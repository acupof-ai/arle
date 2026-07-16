# Agent-RL unified infra — one process, one algorithm seam, one metrics stream

> Status: Active — 2026-07-16. Supersedes the orchestration half of
> [2026-07-13-cc-as-harness-online-opd.md](2026-07-13-cc-as-harness-online-opd.md)
> (its tranches 1-4 remain shipped; its bash loop dies here).

## Verdict

Fold serve + cc rollout + scoring + convert + train + weight-sync into the ONE
`arle train agent-opd` process. Every mechanism already exists in-tree; the bash
loop bypasses all of them:

- The train rollout engine **is** the production serve engine
  (`LoadedInferenceEngine::Cuda` = `ServeInferenceEngine<CudaExecutor, CudaKvPool>`,
  same `cuda_serve_handle` builder as `arle serve`, loaded.rs:1083/:2286).
- Hot weight-sync needs **zero new code**: `sync_lora_from_store →
  remerge_student_lora` re-merges LoRA + flushes the prefix cache in one
  engine-thread closure (serve_engine.rs:342-348) — kills #92 structurally, and
  with it the per-round serve restart and the two per-round ~50 GB reloads.
- Python `score()`/`boot_workdir` duplicate `sandbox.rs` line-for-line and the
  reward denominator has silently diverged (py `n_pass/(n_pass+n_fail)`, ignores
  pytest errors; Rust `passed/len(fail_to_pass)`, counts errors as failures).
- The serial python harness wastes the serve's own `--max-running-requests 4`
  (one request in flight, ever).

Net: delete ≈406 LOC bash/python + ≈1.4k LOC Rust; add ≈450 LOC. User surface
becomes `arle train agent-opd --algo <preset> --rounds N`; everything else
(concurrency, decode config, dense reward, sync cadence, eval cadence, metrics)
is a best-practice default, not a flag to remember.

Adversarially verified (threading / CUDA context / generics / VRAM co-residency
all CONFIRMED): the engine always lives on its own `infer-engine` thread behind
`&self` channel methods; engine+autograd co-residency in one process/context is
existing production (`mem_fraction_static: 0.2`, `--share-frozen-base`,
train_cli.rs:2628-2670). Scope: single-GPU student path (`load_cuda`); DSv4 TP=8
multiproc serve is out of scope.

## A. In-process serve (~150 LOC, 4 files)

1. `coordinator_local_router(serve: ServeHandle<E,K>, …)` →
   `Arc<ServeHandle<E,K>>` (it already `Arc::new`s internally, lib.rs:673);
   wrap the 8 call sites in infer-api/loaded.rs.
2. `ServeInferenceEngine.serve: ServeHandle` → `Arc<ServeHandle>` + a
   `serve_arc()` accessor (serve_engine.rs:22; all uses are `&self`).
3. `LoadedInferenceEngine::local_router(&self, …) -> Result<axum::Router>` —
   CUDA arm clones `Arc<ServeHandle>` + `OpenAiTokenizer` (Clone) into
   `coordinator_local_router`; other arms bail like `remerge_student_lora`.
4. `serve_router_on_thread(router, bind, port, shutdown) -> JoinHandle` — public
   non-blocking twin of the private `bind_and_serve` (infer-api/serve.rs:304-330),
   own tokio runtime on a spawned thread (precedent: train/src/server.rs:48-74).
5. agent-opd wiring: `--serve-port/--serve-bind`, `set_messages_dump_dir`
   (already re-exported, infer-api/lib.rs:71), `shutdown.request()` at loop end.
6. **Quiesce gate**: writeback releases the engine KV pool
   (infer_student.rs:67-119) — a live HTTP request during writeback hits a
   dropped pool, not just stale KV. The driver owns all `claude` subprocesses,
   so quiesce = all children exited + `counters().active_requests == 0`
   belt-and-braces poll.
7. Serve-shaped engine config (slots/pages/max-running-requests for 17K-prompt
   4-way cc traffic vs student scratch) is the one **measured on-pod** trade;
   decode-graph on/off under co-residency decided by that measurement, not
   assumption.

## B. Rust cc harness + unified round loop (~150 LOC new)

Port `cc_attempt` + pass@k aggregation to `crates/train/src/cc_harness.rs`:
spawn `claude -p` via the **spawner** path (mandatory: CUDA-resident
multi-threaded parent, sandbox.rs:148-162), N samples concurrent up to
`max_running_requests`. Boot/score via existing `sandbox.rs::{boot_workdir,
score_workdir}` (single reward definition; the py denominator drift dies).
Convert via `cc_convert` lib in-memory (`Vec<CcRecord>`, no records.jsonl
handoff); keep `--dump-messages-dir` time-window attribution as the first fold,
per-attempt request tagging later.

Round loop = the existing `run_agent_opd_impl` skeleton (eval cadence, round-0
baseline, checkpoint saves, metrics sink — train_cli.rs:2870-3101) with the
rollout arm swapped to the cc harness.

**Sync cadence knob** (the on-policy dial): `--sync every-group | every-round`,
default `every-group` — after each prompt's K-sample group trains, re-merge +
flush, next group rolls under the updated policy. Cost = one shared-prefix
re-prefill per group (cache flushed by design); field norm (2026): bounded
staleness + IS correction, LoRA-only sync per batch is standard.

## C. `UpdatePreset` algorithm seam (replaces the closed `UpdateStrategy` enum)

Six orthogonal fields; an algorithm is a **const value**, not a new enum arm:

```rust
pub struct UpdatePreset {
    pub filter: SampleFilter,      // PassOnly | KeepAll | DropZeroAdvGroup | drop-truncated
    pub advantage: Advantage,      // None | Mean{scope: Batch|Group, std_norm} | ValueGae{gamma, lam}
    pub ratio: RatioGrain,         // None | PerToken | PerSequence
    pub clip: ClipForm,            // HardGate{lo,hi} | SoftClamp{lo,hi}   (weights stay detached)
    pub agg: Aggregation,          // PerSeqTokenMean | GlobalTokenMean{norm_const}
    pub kl: Option<KlReg>,         // {coef, reference: RolloutPolicy /* Teacher later */}
}
```

Presets: `REJECTION_CE`, `SAO_DIS`, `SAO_VALUE`, `GRPO`, `DAPO`, `DR_GRPO`,
`GSPO`, `CISPO`. `RolloutNeeds` derives mechanically (`rollout_logprobs = ratio
!= None`, `keep_failing = filter != PassOnly`). `--update-strategy` stays as a
preset alias.

Three genuine plumbing pieces (everything else is data):
1. **Group identity through the update** — `ScoredTrajectory.group_id`; the
   online path flattens groups at agent_opd.rs:453 while replay already groups
   per task (train_cli.rs:2143-2150). Unlocks per-prompt-group baselines +
   DAPO dynamic-sampling filters.
2. **Hoist the gate/weight builder out of the fused op** — ratio/gate is
   computed inside `fused_linear_pg_loss_indexed` (fused_linear_distill.rs:
   799-803, 931-943); move to a caller-side per-token weight builder keyed on
   `ratio × clip`. Unlocks GSPO's seq-scalar broadcast and CISPO's clamped
   weight; the op keeps its detached-weight contract (PPO grad-through-ratio
   stays structurally excluded — CISPO-family is the native fit, and GSPO beats
   GRPO +13.3% on agentic benchmarks in 2026 head-to-heads).
3. **Split grad-accumulate from optimizer.step** — `masked_writeback_step`
   steps per trajectory (opd.rs:3271); `GlobalTokenMean` (DAPO token-level
   loss) needs batch-level accumulation before one step.

Collapses on landing: `WritebackLoss::{Ce,Dis}` → one per-token-weight op (CE =
weight `1/N`); the replay CE/GKD fork (train_cli.rs:2333-2395) and the GKD⊕SAO
mutual-exclusion bail die — k3 KL already computed (fused_linear_distill.rs:
944-966), `KlReg{reference: RolloutPolicy}` just adds it to the loss with a
coefficient; `Teacher` reference (= on-policy distillation as a regularizer
swap) is the designed-for follow-up.

## D. Always-on metrics.jsonl (extend the existing `--metrics-out` sink)

Three `kind`s, one append-only stream, env-gating retired
(`ARLE_OPD_LOG_DIS_STATS` / `ARLE_AOPD_PROFILE` fold in; profile table stays as
the human view):

- `update`: strategy, trajectories, tokens_trained, policy_loss, critic_mse,
  kl_rollout, is_ratio_mean/max, clip_frac, adv_mean/std, update_secs.
- `group`: task_id, rewards[], reward_mean/std, zero_variance, passed, edited,
  prompt/completion tokens, rollout_secs, rollout_tok_per_sec.
- `round`: pass@k, reward stats, zero_variance_group_frac, phase_secs breakdown,
  rollout_tok_per_sec, held_out_pass_rate/delta.

A run is judged from three trends without reading logs: learning
(round.held_out_delta), off-policy blow-up (update.kl_rollout + clip_frac +
is_ratio_max), signal starvation (group zero-variance fraction + adv_std).

## E. Deletions

| Item | ~LOC | Replacement |
|---|---|---|
| scripts/cc_opd_loop.sh + cc_run.sh + cc_swe_baseline.py | 406 | unified command (A+B) |
| in-house AgentSession arm: `agent_opd.rs::cuda_rollout`, `SandboxToolExecutor`, 4-tool prompts, `run_agent_opd_eval_pass`, in-house-only plumbing | ≈1160 | cc harness is canonical (user decision 2026-07) |
| `tokens_record_to_pairs` + tests (zero non-test callers) | 88 | none — dead |
| replay CE/GKD fork + GKD⊕SAO bail + dup critic/logprob builds | ≈125 | one grouped loop over `UpdatePreset` |
| `--lora-adapters` as chaining vehicle; `adapters_replay` handoff + symlink dance | 27 | TensorStore persists across rounds in-process; flag re-docs as crash-resume; per-round checkpoints stay |

Keep: `update_strategy.rs` (becomes the preset seam), `cc_convert.rs` lib +
offline CLI, `sandbox.rs` boot/score/spawner, `SweTask` loading,
round-loop skeleton.

## F. Gates (pod, H20)

1. Correct inference across re-merge: needle gate ×3 after a mid-run re-merge
   (covers recurrent-sidecar tier invalidation under the new flush path).
2. Reward parity audit: Rust vs old py scoring on one collected round —
   denominator change is **expected and documented**, per-case diffs reviewed.
3. Wall-clock A/B vs the bash loop, same tasks/samples: reload tax (2× ~50 GB
   loads/round) and serial-rollout tax both eliminated; bench entry with Δ%.
4. `every-group` vs `every-round` A/B: pass-rate + wall cost of the per-group
   prefix re-prefill, before flipping any default beyond `every-group`.

## Tranches

| # | What | Depends |
|---|---|---|
| T1 | In-process serve plumbing (A) | — |
| T2 | UpdatePreset seam + 3 plumbing pieces (C) | — |
| T3 | Rust cc harness + unified round loop + sync knob (B) | T1 |
| T4 | metrics.jsonl kinds (D) | T2 |
| T5 | Deletions (E) — same tranche as replacement lands, no half-states | T3 |
| T6 | Pod gates + bench entry + doc statuses (F) | T3-T5 |

## G. Lossless-performance ceiling (refinement, 2026-07-16)

Under strict on-policy, `rollout_i → train_i → merge_i → rollout_{i+1}` is an
information-theoretic dependency, not an infra defect. Squeezing wall-clock
without losing anything therefore has exactly three moves:

1. **Sample-level pipeline — no group barriers (strictly lossless).**
   Score/convert each sample the moment ITS rollout finishes, not when the
   group does: pytest (≤300 s CPU) hides inside the window where sibling
   samples are still rolling. After each merge, proactively re-prefill the
   shared prefix (cache warm-up rides the merge control message) overlapped
   with next-group workdir boot — the per-group re-prefill tax disappears.
   Eval runs off checkpoints, off the critical path. The critical path
   collapses to `max(sample rollouts) → train → merge`.
2. **Staleness is an instrumented dial, not an architecture fork (ε-measured).**
   `--staleness 0` = strict serial. `--staleness 1` = rollout_{i+1} under π_i
   overlaps train_i; the IS ratio exists precisely to correct this, and with a
   rank-16 LoRA at lr 1e-5 the per-group update is tiny → ratio≈1,
   clip_frac≈0, the truncated-IS bias is directly read off the always-on
   metrics (update.clip_frac + kl_rollout ≈ 0 ⇒ empirically lossless). Same
   code path; the only difference is whether the driver awaits the merge.
   Single-GPU degenerate form of AReaL's decoupled PPO. Precondition: engine
   KV pool + autograd activations co-resident (measured on-pod, gate F.5).
3. **Engine-native trajectory capture (kills three reconstruction errors at
   once).** Record `(request_id, prompt_token_ids, gen_token_ids,
   gen_logprobs)` at generation time, D2H at request finish
   (graph-compatible). Replaces: time-window dump attribution (fragile),
   chat-template re-render for masks (drift risk — span offsets come from the
   engine's own renderer instead), and train-side V0 logprob recomputation
   (whose "θ unchanged since generation" assumption is exactly what breaks at
   staleness>0 — the 2026 "missing old logits" failure). Cost ≈ one f32
   gather/token.

Honest trade in (3): generation-time logprobs carry FP8-serve numerics while
π_θ is computed under bf16 train numerics — ratio ≠ 1 even at θ = θ_b; V0
recomputation is the mirror image (ratio exactly 1, but only valid at
staleness 0). Decide at T3 with one measurement: the ratio distribution gap
between both sources at staleness 0 = the numerics-noise floor.

Added gate: **F.5** — co-residency VRAM ledger (KV pool + activations) before
enabling `--staleness 1`; **F.6** — logprob-source ratio-floor measurement.

## Field grounding (mid-2026 survey)

verl/NeMo-RL converged on **two pure functions over masked token tensors**
(advantage estimator + policy loss, config-dispatched) as THE pluggable surface
— `UpdatePreset` is that shape. GSPO (seq-level ratio) is the winning agentic
variant; DAPO's knobs (clip-higher / dynamic sampling / token-level loss) are
the production recipe; detached-IS-weight designs (CISPO) match our op contract.
Async staleness practice = bounded versions + truncated-IS (our DIS gate) +
background weight publish; LoRA-only sync per batch is standard. OPD enters the
same seam as a KL-reference swap (Thinking Machines / AReaL). Trajectory-level
group advantage broadcast over tokens + tool-output masking remains the
production norm for SWE agents (DeepSWE, Nebius) — turn-level credit is the
research frontier the seam leaves room for (per-token advantage vector already
flows end-to-end).
