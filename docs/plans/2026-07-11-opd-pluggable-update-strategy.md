# OPD pluggable update strategy — architecture + SAO Phase 1

> Status: Active

Make the policy-update algorithm a swappable component. The rollout/scoring
harness is strategy-agnostic; each algorithm is one closed-set variant. First
concrete second strategy: SAO Phase 1 (DIS + binary advantage).

## Why

Today the update is hardwired (`masked_writeback_ce_step`), and the codebase
already has scattered ad-hoc variants (`_frozen_prompt_kv`, `_dispatch`,
`rubric_writeback_ce_step_batched`, `masked_gkd_*`). Adding SAO as another
flagged branch layers old+new paths. Instead: one abstraction, converge the rest.

## Abstraction (enum, not dyn trait)

Closed set selected once by a CLI flag; `Optimizer` is a generic type param, so a
dyn trait would force it behind `dyn`. An enum with static-dispatch `update<O:
Optimizer>` is the lowest-entropy form and still one-variant-to-extend.

```rust
// crates/train/src/update_strategy.rs  (new)
pub struct RolloutNeeds { pub keep_failing: bool, pub rollout_logprobs: bool }

pub struct ScoredTrajectory {
    pub prompt_ids: Vec<u32>,
    pub response_ids: Vec<u32>,
    pub response_mask: Vec<u8>,      // Skip-Observation: 1 = LLM token, 0 = tool
    pub reward: f32,                 // pytest: 1.0 pass / 0.0 fail
    pub rollout_logprobs: Option<Vec<u32>>, // per response token, π_rollout; Some iff needs
}

pub enum UpdateStrategy {
    RejectionCe,                              // current behavior (default)
    SaoDis { eps_low: f32, eps_high: f32 },   // Phase 1
}

impl UpdateStrategy {
    pub fn needs(&self) -> RolloutNeeds { /* CE: {false,false}; DIS: {true,true} */ }
    pub fn update<O: Optimizer>(
        &self, batch: &[ScoredTrajectory], student: &Qwen35Model,
        all_params: &[TensorId], trainable: &[TensorId], opt: &mut O,
        vocab: usize, window: usize, store: &mut TensorStore,
    ) -> Result<f32>;
}
```

- `RejectionCe::update` = filter `reward > 0`, per trajectory `masked_writeback_ce_step`, mean.
- `SaoDis::update` = advantage `A_i = reward_i − mean(reward)`; per trajectory
  `masked_writeback_dis_step(.., rollout_logprobs, A_i, eps_low, eps_high)`, mean.

## Pieces (ordered, cheapest+isolatable first)

1. **Autograd — weighted PG loss** (`crates/autograd/src/ops/fused_linear_distill.rs`):
   new `fused_linear_pg_loss_indexed`, a per-token-weighted sibling of
   `fused_linear_ce_loss_indexed`. Extra args: `rollout_logprobs: &[f32]` (per
   position), `advantage: f32`, `eps_low`, `eps_high`. Per position: compute
   `logp = logπθ(target)` (already computed for CE); `r = exp(logp − rollout_lp)`;
   `gate = (1−eps_low < r < 1+eps_high) as f32`; weight `w = advantage · gate`
   (detached scalar); `loss += −w·logp`; `grad_student[pos] = w · (softmax −
   onehot)` (CE grad scaled by `w` instead of the uniform `1/N`). r/gate/advantage
   are weights only — gradient flows through `logp` (REINFORCE form).
   **Self-check** (`#[test]`): `advantage = 1/N`, `rollout_lp = logπθ` (→ r=1,
   gate=1, w=1/N) must reproduce `fused_linear_ce_loss_indexed` bit-for-bit.

2. **opd.rs — DIS step + rollout-logprob capture**:
   - `masked_writeback_dis_step<O>` = clone of `masked_writeback_ce_step` with the
     fused call swapped to `fused_linear_pg_loss_indexed`; everything else
     (forward, backward, optimizer, seq-adaptive offload) identical.
   - `capture_rollout_logprobs(student, prompt_ids, response_ids, response_mask,
     store) -> Vec<f32>`: tape-OFF forward over prompt++response, gather
     `log_softmax(logits)[target]` at each response position. This is π_rollout at
     V0 — called in the harness before any optimizer step, so θ is still the
     rollout policy.

3. **update_strategy.rs** — the enum + structs + two impls above.

4. **agent_opd.rs — strategy-driven harness**:
   - `let needs = strategy.needs();`
   - remove the hardcoded `if !passed continue` (`:600`); push EVERY trajectory
     with `reward` into `Vec<ScoredTrajectory>` when `needs.keep_failing`, else
     passing-only (current behavior).
   - after collection, before the update: if `needs.rollout_logprobs`, fill each
     trajectory's `rollout_logprobs` via `capture_rollout_logprobs` (θ = V0).
   - replace the writeback loop (`:639`) with `strategy.update(&batch, ..)`.

5. **CLI** (`args.rs` + `runtime_flags.rs`): `--update-strategy rejection-ce|sao-dis`
   (default `rejection-ce`), `--sao-eps-low` (0.8), `--sao-eps-high` (3.0).

## Invariants / gate

- Default `rejection-ce` is **byte-identical** to today (same filter, same
  `masked_writeback_ce_step`, no logprob forward paid).
- Correctness gate for `sao-dis`: needle / self-consistency + one round with
  finite, non-exploding loss on the 73-task real corpus. NOT token-identity.
- SAO Phase 2 (value critic + Skip-Obs GAE) plugs in as a third variant later —
  no harness change.

## Cost note

`sao-dis` pays one extra tape-off forward per trajectory (π_rollout capture,
~forward-phase cost) + trains on all trajectories (more writeback steps than the
passing-only default). Bench entry required per the runtime-change rule.
