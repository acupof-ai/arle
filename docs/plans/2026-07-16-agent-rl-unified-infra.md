# Agent-RL unified infra — implementation plan (v2, re-ranked)

> Status: Active — 2026-07-16 (v2 full rewrite; v1 sections A-H consolidated).
> Supersedes the orchestration half of
> [2026-07-13-cc-as-harness-online-opd.md](2026-07-13-cc-as-harness-online-opd.md).
> Landed so far: **P0** in-process serve plumbing (`b5f0f406`). **P1** UpdatePreset
> in flight.

## Verdict

One `arle train agent-opd` process = serve + cc rollout + scoring + convert +
train + weight-sync. User surface: `--update-strategy <preset> --rounds N`;
everything else is a measured best-practice default. Deletes ≈406 LOC
bash/python + ≈1.4k LOC Rust, adds ≈600.

Ranking driver (survey-grounded): **rollout is ~96% of a LoRA-RL run** (Unsloth
FP8+LoRA measurement; Arnal >80% fleet-wide); sync batching idles at 20-40% GPU
util vs ~90% achievable; **51-66% of GRPO groups are zero-variance** (full
rollout cost, zero gradient). So priorities are: orchestrator concurrency +
CPU pipeline first, sample-efficiency second, staleness overlap third,
train-step micro-opts ~never.

Wall-clock estimate vs today's serial bash loop (estimates, each gated before
credit is claimed): concurrency+pipeline ~2-2.5×, task-selection 2-6×
*effective* compute, staleness-1 ~1.3×, spec-decode 1.3-2×. Compounding
plausibly ≥5× to a given capability level.

## Architecture (final form)

```
arle train agent-opd --update-strategy dapo --rounds 8
┌──────────────────────────────────────────────────────────────────┐
│ driver (round loop, fixed)                                       │
│  boot pool ─▶ cc-run pool ─▶ score pool ─▶ group asm ─▶ trainer  │
│  (bounded std::mpsc queues; sample-level flow, no group barrier) │
│                 │ claude -p ×K ──HTTP──▶ serve thread            │
│                 ▼                                                │
│  UpdatePreset.update ─▶ remerge/adapter-swap ─▶ ControlMessage   │
│                                                                  │
│ serve thread: axum OpenAI+Anthropic router over Arc<ServeHandle> │
│ engine thread: Engine<CudaExecutor,KvPool> (sched+Radix+batching)│
│ autograd student: LoRA rank16, shares frozen FP8 base pointers   │
└──────────────────────────────────────────────────────────────────┘
   metrics.jsonl (update / group / round)  ← every stage feeds it
```

Three async invariants (published convergence, single-GPU form):
1. token stream never stalls for weight sync (sub-second adapter sync via
   engine-thread control closure);
2. training never waits on CPU/env time (boot/score/convert pipelined
   off-path);
3. a group never waits on its slowest member for scoring (sample-level flow;
   the *update* waits for group rewards — information-theoretic, not infra).

Two pluggable seams, everything else fixed:
- **Algorithm** = `UpdatePreset` (data). New algorithm = new preset value.
- **Rollout harness** = subprocess driver (`CcHarness`). New harness = new
  driver; serve/train/sync untouched.

Scope: single-GPU student path (`load_cuda`); DSv4 TP=8 out of scope.

## Concurrency × on-policy coupling (design fact, decided upfront)

- `--staleness 0` (default): task groups are **sequential**; in-flight
  concurrency = K samples of the current group. Raise K (default 4 → 8 gated
  on F.5 KV budget) — bigger K is the staleness-0-compatible concurrency
  lever, and cuts zero-variance probability (p⁴+(1-p)⁴ → p⁸+(1-p)⁸: 0.66 →
  0.43 at p=0.9; dense reward cuts it further).
- `--staleness 1`: M groups in flight; merge after each group trains;
  other in-flight groups continue under their behavior adapter (whole
  trajectory version-tagged; token-TIS corrects). This is where the 2.4-2.7×
  published agentic gains live. Requires P6 (generation-time behavior
  logprobs) — **not before**.
- Boot + prefix warm-up of group i+1 overlap group i's train/merge in BOTH
  modes (CPU + cache-warm only — staleness-free).

## Phases (re-ranked)

### P0 — in-process serve plumbing ✅ (`b5f0f406`)

`coordinator_local_router(Arc<ServeHandle>)`, `ServeInferenceEngine.serve:
Arc<_>` + `serve_arc()`, `LoadedInferenceEngine::local_router()`,
`serve_router_on_thread() -> ServeThread` (owns `ServeShutdown`), re-exports.
132+/13−, all lanes green.

### P1 — `UpdatePreset` algorithm seam (in flight)

Six orthogonal fields, presets as values:

```rust
pub struct UpdatePreset {
    pub filter: SampleFilter,   // PassOnly | KeepAll | DropZeroAdvGroup | DropTruncated
    pub advantage: Advantage,   // None | Mean{scope: Batch|Group, std_norm} | ValueGae{gamma, lam}
    pub ratio: RatioGrain,      // None | PerToken | PerSequence
    pub clip: ClipForm,         // HardGate{lo,hi} | SoftClamp{lo,hi}   (weights detached)
    pub agg: Aggregation,       // PerSeqTokenMean | GlobalTokenMean{norm_const}
    pub kl: Option<KlReg>,      // {coef}; reference = rollout policy; Teacher slot later
}
```

Presets: `rejection_ce, sao_dis, sao_value, grpo, dapo, dr_grpo, gspo, cispo`.
`RolloutNeeds` derives (`rollout_logprobs = ratio != None`, `keep_failing =
filter != PassOnly`); `needs_value_critic = advantage == ValueGae`.
Three plumbing pieces: `ScoredTrajectory.group_id`; gate/weight builder hoisted
out of `fused_linear_pg_loss_indexed` (`WeightForm{HardGate,SoftClamp,
Precomputed}`; KL folds in as `coef×(r−1)` added to the detached weight;
GSPO = capture pass → seq-ratio → `Precomputed`); grad-accumulate/step split in
`masked_writeback_step` for `GlobalTokenMean`. Behavior contract: the three
shipped strategies byte-identical through their presets; default stays
`rejection-ce`. Note: our HardGate double-sided detached gate is the same
family as IcePop's [0.5,5] mask — the correction that covers BOTH async
staleness and FP8-serve/bf16-train numerics mismatch (measured token KL ~1e-2
elsewhere; uncorrected quantized rollout collapses training).

### P2 — unified orchestrator, async-native internals (~350 LOC new)

`crates/train/src/cc_harness.rs` + round-loop rewiring in `train_cli.rs`
(reusing the existing skeleton at train_cli.rs:2870-3101: eval cadence, round-0
baseline, checkpoint saves, metrics sink).

**Driver = 3-stage bounded pipeline, house-style threads + std::mpsc** (no
tokio in train; matches engine/control-plane precedent):
- *boot pool* (2 workers): `sandbox.rs::boot_workdir` (staged copytree + git
  init) — pre-boots next group's K workdirs during current group's
  train/merge.
- *cc-run pool* (width = serve `max_running_requests`): spawn `claude -p
  --model … --allowedTools "Bash Read Write Edit Grep Glob" --output-format
  json --dangerously-skip-permissions` via **spawner** (mandatory:
  CUDA-resident multithreaded parent, sandbox.rs:148-162; IS_SANDBOX=1 env);
  per-sample `(t_start_ms, t_end_ms)` recorded for dump attribution;
  `--cc-timeout` 600s.
- *score pool* (2-4 workers, CPU-bounded): `sandbox.rs::score_workdir` the
  moment a sample's cc exits — pytest (≤300s) hides inside siblings' rollout
  window. Single reward definition (Rust semantics: errors count as failures,
  denominator = len(fail_to_pass); the py drift dies — documented Δ in the
  first bench entry). Tiered scoring: fail_to_pass first, early-exit on
  first F2P failure, pass_to_pass only on F2P success, ~120s soft cap.
- *group assembler*: `HashMap<task_id, Vec<Scored>>`; a group completes at K
  scored samples → cc-convert (lib call, in-memory `Vec<CcRecord>`, time-window
  attribution kept for this fold) → trainer.
- *trainer*: capture logprobs at V0 → `UpdatePreset.update` → sync.

**Weight sync**: `sync_lora_from_store → remerge_student_lora` (atomic
re-merge + `invalidate_prefix_cache` in one engine-thread closure). Quiesce =
all cc children exited + `counters().active_requests == 0` poll (needed beyond
staleness: writeback drops the engine KV pool — a live request would hit a
dropped pool). Post-merge: **prefix warm-up** — enqueue one prefill of the
shared system prompt, overlapped with next-group boot (kills the per-group
re-prefill tax).

**metrics.jsonl** (moved into this phase — it is the sensor layer P4 gates and
P5 scheduling read). Extend the existing `--metrics-out` JsonlSink
(train_cli.rs:3057) with `kind` rows, always-on, env-gates retired
(`ARLE_OPD_LOG_DIS_STATS`, `ARLE_AOPD_PROFILE` fold in; human table stays):
- `update`: preset, trajectories, tokens_trained, policy_loss, critic_mse,
  kl_rollout, is_ratio_mean/max, clip_frac, adv_mean/std, update_secs.
- `group`: task_id, rewards[], reward_mean/std, zero_variance, passed, edited,
  prompt/completion tokens, rollout_secs, rollout_tok_per_sec.
- `round`: pass@k, reward stats, zero_variance_group_frac, phase_secs,
  rollout_tok_per_sec, held_out_pass_rate/delta.
Run judgment without logs: learning (round.held_out_delta), off-policy blow-up
(update.kl_rollout + clip_frac + is_ratio_max), signal starvation
(zero_variance_group_frac + adv_std), throughput (phase_secs, tok/s).

**CLI surface (full)**: `--rounds`, `--update-strategy <preset>`, `--dataset
--staged-root --eval-dataset`, `--samples-per-prompt` (default 4; 8 after
F.5), `--sync every-group|every-round` (default every-group), `--serve-port
--serve-bind`, `--cc-timeout 600`, `--eval-every 2`, existing LoRA/lr flags.
Serve-shaped engine config derived: `num_slots = samples_per_prompt`,
`total_pages = slots × 22K/16 + headroom` (formula, not magic constant);
decode-graph on/off decided by F.5 measurement under co-residency.

### P3 — deletions (land in the same tranche as each replacement)

| Item | ~LOC | Replacement |
|---|---|---|
| scripts/cc_opd_loop.sh + cc_run.sh + cc_swe_baseline.py | 406 | P2 |
| in-house AgentSession arm (`agent_opd.rs::cuda_rollout` ~590, `SandboxToolExecutor` ~220, 4-tool prompts ~66, `run_agent_opd_eval_pass` ~56, in-house-only plumbing ~230) | ≈1160 | cc harness canonical |
| `tokens_record_to_pairs` + tests (zero non-test callers) | 88 | dead |
| replay CE/GKD fork + GKD⊕SAO bail + dup critic/logprob builds | ≈125 | P1 collapsed them |
| `--lora-adapters` as chaining vehicle; `adapters_replay` handoff | 27 | TensorStore persists in-process; flag re-docs as crash-resume |

Keep: `update_strategy.rs` (the seam), `cc_convert.rs` lib + offline CLI,
`sandbox.rs` boot/score/spawner, `SweTask` loading, round-loop skeleton,
`--replay-records` as offline entry.

### P4 — pod validation gates (H20, GPU 0/1/7, pinned dir)

- **F.1** correct inference across sync: needle gate ×3 after a mid-run
  re-merge (also covers recurrent-sidecar tier invalidation).
- **F.2** reward parity audit: Rust vs old py scoring on one collected round;
  denominator change expected + documented, per-case diffs reviewed.
- **F.3** wall-clock A/B vs the bash loop, same tasks/K: reload tax + serial
  rollout eliminated; wins/ entry with Δ%.
- **F.4** `every-group` vs `every-round` A/B: pass-rate + wall cost.
- **F.5** co-residency VRAM ledger: KV pool (K=4 vs 8 × 22K) + autograd
  activations + decode-graph on/off; sets K default and page formula headroom.
- **F.6** logprob-source ratio floor: V0-recompute vs generation-capture ratio
  distribution at staleness 0 = numerics-noise baseline (FP8 serve vs bf16
  train); informs P6/P7.
- **F.7** re-merge requant drift: folding bf16 LoRA into FP8 base each sync is
  dequant→add→requant — measure merged-vs-adapter-separate logit gap over N
  syncs. If material → **adapter-separate serving** (rank-16 LoRA GEMM at
  inference; sync = true sub-ms adapter swap; also enables KV-keep across
  syncs per PipelineRL's stale-KV≈recompute measurement + TIS).

### P5 — task-selection scheduler (2-6× effective compute; cheap; after P2+P4)

Reads per-task history from metrics.jsonl `group` rows; lives in the driver's
task iterator, ~80 LOC:
- **zero-variance skip** (GRESO-style): P(skip) grows with consecutive
  zero-variance rounds for that task; floor ε=0.1 re-explore.
- **comfort band**: sample tasks ∝ proximity to ~50% EMA pass-rate
  (SPEED-RL/DOTS: intermediate difficulty maximizes gradient SNR).
- **retirement**: EMA pass ≥0.9 over 3 rounds → retire (Polaris practice).
- All three are *scheduler* policy — `UpdatePreset.filter` stays the loss-side
  guard. Emit skipped/retired counts in `round` rows (no silent truncation).

### P6 — engine-native trajectory capture (correctness + staleness enabler)

Engine records per-request `(request_id, prompt_token_ids, gen_token_ids,
gen_logprobs)`: sampler-site gather of the chosen token's logprob into a
device ring, D2H at request finish (graph-compatible — no per-step sync).
Dump sink keyed by request_id replaces time-window attribution; masks come
from the engine's own span renderer (`render_structured_chatml_with_spans`)
instead of client-side re-render. Kills: window fragility, re-render drift,
and the V0-recompute "θ unchanged" assumption (the published missing-old-logits
failure). Polar (arXiv 2605.24220) validates the exact shape: RL through
unmodified Claude Code via proxy-recorded token ids + logprobs, +4.8
SWE-Bench-Verified on a 4B. Decision input: F.6 ratio floor.

### P7 — staleness dial (needs P6)

`--staleness 0|1`. At 1: driver admits group i+1 before group i's merge;
trajectories carry `behavior_version`; ratio uses generation-time logprobs;
existing HardGate/TIS corrects (published envelope: 1-2 steps safe everywhere;
AReaL η≤8 within 1%). KV-keep across adapter swaps becomes an option per F.7.
Loss is *measured*, not assumed: clip_frac + kl_rollout ≈ 0 ⇒ empirically
lossless.

### P8 — gated roadmap (RE-COSTED 2026-07-16: substrate exists in-tree, these
are wire-and-license, not build)

- **Spec decode in rollout** — AUDITED 2026-07-16, two license paths:
  - **(a) zero-code, license-ready**: the DSpark lane already implements EXACT
    rejection sampling (`u < p/q` + residual `max(0,p−q)` draw,
    sampling.cu:844-857) and already routes SAMPLED decoding through spec
    (`dspark_accept_commit_sampled`, executor/qwen35.rs:1857-1871). At temp=1
    top_p=1 acceptance is exact vs the policy; with top_p<1 exact vs the
    nucleus-filtered policy (the standard RL sampler). Needs only a Qwen3.6
    drafter checkpoint (`--spec-type dspark --mtp-draft-model <dir>`).
  - **(b) ~200-250 LOC port** if the checkpoint-native NextN-MTP head must be
    used instead: that lane is greedy-only AND sampling-unreachable today
    (temp≠0 falls back to no-spec, executor/qwen35.rs:1486-1495; argmax verify
    qwen35.rs:3932); port the shipped DSpark rejection twin onto it — zero new
    CUDA, buffer-rollback coverage carries over unchanged.
  Gate either way: needle ×3, acceptance rate, identical reward curve on one
  round. 1.3-2× on the dominant cost.
- **Experience replay** — substrate: `--replay-records` + `--replay-epochs`
  already exist. Gap: driver-side retention policy (age ≤10 steps,
  fresh-anchored, |A|-prioritized) + IS-corrected reuse (the preset's ratio
  machinery already covers it — behavior logprobs are carried). Gate: held-out
  parity at reuse 2-3×.
- **Privileged self-distillation for all-fail tasks** (HDPO/OPSD) — substrate:
  GKD machinery complete (`KlDirection` fwd/rev, temperature, λ-mix, teacher
  source flag, teacher surface on infer-api). Gap: a privileged-prompt teacher
  lane (same model + failing pytest output / reference patch in context) +
  `KlReg{reference: Teacher}` wiring. The ~10×-class move; rescues the
  zero-gradient tail. Gate: all-fail-task pass-rate delta on the real corpus.

Standing rule this re-cost enforces: **audit in-tree substrate before costing
any lever** (先用最好的再自己写) — the v1 costing of all three was wrong in the
expensive direction.

## Priority rationale (value × effort × dependency)

| Rank | Phase | Expected effect | Effort | Depends |
|---|---|---|---|---|
| 1 | P1 preset seam | algorithm iteration cost → ~0; TIS machinery | in flight | — |
| 2 | P2 orchestrator+pipeline+metrics | 2-2.5× wall vs bash loop; observability | ~350 LOC | P0 ✓ |
| 3 | P3 deletions | −1.8k LOC, no half-states | mechanical | P2 |
| 4 | P4 gates | licenses defaults; measures F.5/F.6/F.7 unknowns | pod runs | P2-P3 |
| 5 | P5 task selection | 2-6× effective compute; biggest single multiplier | ~80 LOC | P2 metrics |
| 6 | P6 engine capture | correctness hardening; unlocks P7 | engine plumbing | F.6 |
| 7 | P7 staleness 1 | ~1.3-1.7×, ε-measured | small | P6 |
| 8 | P8 spec-decode license / replay policy / priv-distill lane | 1.3-2× / 1.4× / ~10×-class | wire-and-license (substrate in-tree; see P8 re-cost) | spec+replay: P4; distill: P7 |

Re-ranking vs v1: metrics moved INTO the orchestrator phase (sensor layer,
not an afterthought); task selection promoted above staleness (bigger
multiplier, smaller effort, zero risk); engine capture promoted from
"later" to the P7 gatekeeper (Polar-validated); train-step micro-opts
(packing, recompute tuning) explicitly DROPPED (<4% share).

## Defaults contract (run it and it's best practice)

| Knob | Default | Why |
|---|---|---|
| preset | rejection-ce (flip after P4 A/B) | current validated baseline |
| samples_per_prompt K | 4 → 8 after F.5 | staleness-0 concurrency + zero-variance cut |
| sync | every-group | user's on-policy requirement; F.4 prices it |
| staleness | 0 (dial exists after P7) | strict on-policy until ε is instrumented |
| scoring | tiered F2P-first, 120s soft cap, Rust semantics | reward wall −; single definition |
| prefix warm-up after merge | on | hides re-prefill tax |
| metrics.jsonl | always-on | no env-gate; agents parse runs |
| task selection | on after P5 (skip/band/retire, ε=0.1) | 2-6× effective |
| decode-graph / KV pages | F.5 measurement | co-residency budget, not assumption |

## Field grounding (kept from v1, abridged)

verl/NeMo-RL: algorithm surface = advantage-estimator + policy-loss as
config-dispatched data — `UpdatePreset` is that shape. GSPO wins agentic
head-to-heads (+13.3% vs GRPO, ARLArena). Async convergence: bounded staleness
1-2 + truncated token-IS; in-flight weight swap with kept KV ≈ recompute
(PipelineRL Fig.7); LoRA adapter-only sync is sub-ms in 8/16 libraries.
Harness-in-the-loop is published practice (Polar, KAT-Coder-V2.5, Agent
Lightning, verifiers-v1, Cursor Composer). Rollout waste: zero-variance 51-66%
of groups; GRESO 2.0×, AERO 1.8-1.9×, SPEED-RL 2-6×. OPD 9-30× vs RL
(Thinking Machines; Qwen3 ~10×); HDPO/OPSD = privileged self-distill without an
external teacher. Trajectory-level group advantage + tool-output masking remains
the SWE production norm (DeepSWE, Nebius); turn-level credit is the research
frontier the per-token advantage vector leaves room for.
