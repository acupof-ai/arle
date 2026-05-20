# OPD rollout `forward_last_logits` — correctness landed, wall-clock pending — AMD Ryzen 7 3700X, 2026-05-20

> **Status:** `pending-bench`. Correctness gates green; wall-clock A/B at the
> Qwen3-0.6B production shape was OOM-killed by the kernel (cooperative
> session memory pressure with peer codex agent). Moderate shape A/B is
> within noise. Codex owns the production-shape verification run.

## Goal

- **(optimization)** OPD rollout student forward currently computes `lm_head`
  over all `seq_len` positions, but `greedy_argmax_last_row` only reads the
  last row. Route the rollout through a new `forward_last_logits` that
  slices the post-`final_norm` hidden to the last position before applying
  `lm_head`. Expected to save `(seq_len - 1) × hidden × vocab` FMAs per
  rollout iteration — at Qwen3-0.6B (`vocab=151_936`) this is the dominant
  rollout cost based on [`2026-05-19-bench-cpu-matmul-mixed-dispatch-qwen3-06b.md`](2026-05-19-bench-cpu-matmul-mixed-dispatch-qwen3-06b.md)
  showing `lm_head` at 54 % of the per-step matmul budget.
- Maintain bit-identical OPD step output (verified via `test_opd_determinism`).

## Hypothesis (Qwen3-0.6B production shape, **NOT YET WALL-CLOCK VERIFIED**)

At Qwen3-0.6B (`hidden=1024`, `vocab=151_936`) with `prompt_len=3`,
`rollout_len=2`:

| Quantity | Full lm_head | Last-row lm_head | Saved |
|---|---:|---:|---:|
| lm_head FMAs, rollout iter 0 (seq=3) | 3 × 1024 × 151_936 = 4.67e8 | 1 × 1024 × 151_936 = 1.56e8 | 3.11e8 |
| lm_head FMAs, rollout iter 1 (seq=4) | 4 × 1024 × 151_936 = 6.22e8 | 1 × 1024 × 151_936 = 1.56e8 | 4.67e8 |
| **Total rollout lm_head FMAs saved** | — | — | **7.78e8** |

At [`2026-05-19-bench-cpu-matmul-mixed-dispatch-qwen3-06b.md`](2026-05-19-bench-cpu-matmul-mixed-dispatch-qwen3-06b.md)'s
measured 8.55 GFLOPs/s `lm_head` rate (the bandwidth-bound `lm_head` lane),
7.78e8 FMAs ≈ **91 ms saved per OPD step**. Total OPD step matmul post-`15fa6cf`
was 1.80 s, so the projected step-level saving is **~5 %** — smaller than the
naïve "29 % of lm_head, lm_head is 54 % of step → 16 %" framing because the
teacher + student-full forwards (the other two lm_head invocations) still
hit every position for KL loss. The win is restricted to the rollout
sub-step only.

Why this didn't show up in the moderate-shape A/B is the open question
(§Problems).

## Command (correctness gates)

```bash
cargo build -p train --release
cargo test -p train --release --test test_opd_determinism --test test_opd_grad_check
cargo clippy -p train --all-targets --release -- -D warnings
cargo run -p train --example rollout_last_logits_ab_bench --release \
  | tee bench-output/2026-05-20-rollout-last-logits-ab/run1.txt
# Production-shape run pending; codex owns this verification step.
```

## Environment

| Item | Value |
|---|---|
| Backend | CPU (`autograd::backend`; lazy device handles disabled — pure host path) |
| CPU | AMD Ryzen 7 3700X 8C/16T, Zen 2 |
| OS / arch | Linux x86_64 |
| Rust | `rustc 1.95.0 (59807616e 2026-04-14)` |
| Substrate | `15fa6cf` (mixed-dispatch sgemm) + this tranche |
| Cooperative state | Codex parallel session active; **both can't run heavy concurrently** (2026-05-20 SIGKILL observed at exit 137) |
| Feature set | `cargo run -p train --example rollout_last_logits_ab_bench --release` |

## Results — correctness gates only

| Gate | Result |
|---|---|
| `cargo test -p train --release` | 21 tests passed |
| `cargo test -p train --test test_opd_determinism --release` | `opd_step_same_prompt_seed_and_lr_is_bit_identical` ok |
| `cargo test -p train --test test_opd_grad_check --release` | `kl_distill_loss_student_logits_grad_matches_finite_difference` ok |
| `cargo clippy -p train --all-targets --release -- -D warnings` | clean |
| A/B equivalence (rollout tokens) | identical at moderate shape: `[7111, 23273, 15788, 12324]` |

The bit-identical OPD-step loss is the load-bearing correctness pin: a buggy
slice would change the rollout token sequence and the loss would diverge.
PASS means the new path matches the prior full-lm_head + last-row argmax to
the bit.

## Results — moderate-shape A/B (vocab=32_768, hidden=512, layers=6, rollout_len=4)

```
full lm_head (baseline)  mean=4.959291s median=4.988655s sigma=0.146877s sigma_pct=2.962%
last-row lm_head         mean=4.981714s median=4.942449s sigma=0.057228s sigma_pct=1.149%

rollout speedup mean_full/mean_last = 0.995x
rollout saved per iteration: mean -0.005606s (4 iterations)
```

Within noise; **no measurable wall-clock win**. The 1.149 % σ on the new
path is materially lower than the baseline's 2.962 %, which is suggestive
(smaller intermediate logits = less store churn = lower variance) but does
not constitute a speed win.

## Results — production-shape A/B (vocab=151_936, **PENDING — codex owns**)

Pending. The dev-box wall-clock A/B at Qwen3-0.6B-vocab shape SIGKILL'd at
exit 137 (OOM) when both Claude and codex were running. Per the user's
cooperative directive (2026-05-20), the production-shape verification run
is delegated to codex. Expected harness command (codex's tmux session,
Claude-side idle):

```bash
cargo run -p train --example rollout_last_logits_ab_bench --release \
  | tee bench-output/2026-05-20-rollout-last-logits-ab/run2-qwen3-0_6b-vocab.txt
```

Codex may need to: reduce `MEASURED_RUNS`, add `retain_ids` pruning between
rollout iterations (the bench currently accumulates rollout activations
across all 5 measured × 2 variants = 40 forwards, peak ~1.5 GB), or split
the A/B into separate processes. Those are complex implementation choices —
left for codex per the work split.

## Problems

- **Moderate-shape framing trap (§0 SOLID self-check).** The moderate-shape
  zero-win result is real evidence — but at vocab=32 768 the `lm_head`
  matmul is small enough that it shares the per-iter time roughly equally
  with the transformer body; saving 75 % of a 30 % slice yields ~22 %, well
  within the 3 % σ noise floor compounded over the per-iter variance.
  Cannot conclude "optimization doesn't work" from this shape alone. The
  production shape (vocab=151 936) shifts `lm_head` to ~50 % of step;
  saving most of it should be visible above noise.
- **Possible confounder — `lm_head` transpose copy.** `linear_forward`
  (`crates/train/src/qwen35.rs:980-1023`) calls `transpose(weight, 0, 1)`
  every invocation. `transpose_host_eager`
  (`crates/autograd/src/ops/layout.rs:181`) is a physical data copy. At
  Qwen3-0.6B the `lm_head` weight is `[151_936, 1024] × 4 B = 623 MB`, so
  every forward physically reallocates and copies 623 MB. At ~10 GB/s host
  bandwidth that's ~60 ms per call — possibly the dominant cost, and it's
  paid identically by both A/B variants, so the rollout-savings win cannot
  appear above it in wall-clock. **This is the next-tier perf axis** (see
  [`docs/research/lm-head-transpose-cache.md`](../../research/lm-head-transpose-cache.md)
  — sibling research entry filed alongside this tranche).
- **Cooperative OOM at production shape.** Running the bench with both
  Claude and codex active hit a hard OOM on the 31 GB dev box. The bench
  itself peaks at ~1.5 GB but the workspace's compounded RSS exceeded
  available memory. Codex's session was SIGKILL'd. Lesson: bench
  scheduling must be exclusive when both agents run.

## Learnings

1. **Hypothesis-chain correctness ≠ wall-clock win.** The FLOP count is
   correct, the slice fires in the right place, determinism holds — and
   wall-clock is still zero at moderate shape. Per §0 SOLID, hypothesis ≠
   evidence: I cannot claim the optimization "works" until a production-shape
   A/B shows it above the noise floor. The artefact is correct; whether
   it's a win depends on numbers I don't have.
2. **Don't burn shared compute on hypothesis-fishing.** With codex running
   concurrently, every speculative bench risks OOM-killing the peer. The
   right move when the moderate result was zero was *stop benching and
   research the deeper bottleneck* (transpose copy), not retry at bigger
   shape. This entry captures that pivot.
3. **Cooperative `pending-bench` is a valid status.** The CLAUDE.md
   "regression-check minimum" + "if the bench can't run locally" rules
   permit landing a stub wins entry with the verification ticketed. With a
   peer agent doing verification, "can't run locally without OOM-ing
   peer" is structurally equivalent to "can't run on this machine — needs
   remote box." The stub is the honest artefact.

## Rule

When a wall-clock A/B at one shape is within noise, do not extrapolate
"the optimization works at the production shape" without measuring. Either
ship the change as `pending-bench` with the projection explicit, or kill
the branch. Do not silently project moderate → production.

When running benches in a cooperative-agent workspace, peek tmux for the
peer's RSS before starting any large run. If the peer is active and the
combined budget could exceed available memory, **do not start the bench**
— hand it to the peer or queue it for serial execution.

## Artefacts

- Bench source: `crates/train/examples/rollout_last_logits_ab_bench.rs`
- Code change: `crates/train/src/qwen35.rs` (added `forward_last_logits`,
  refactored `forward_batch_indices` to share a `forward_hidden_batch_indices`
  helper), `crates/train/src/opd.rs` (rollout call site + greedy-argmax
  helper rename).
- Raw moderate-shape: `bench-output/2026-05-20-rollout-last-logits-ab/run1.txt`
- Companion research: `docs/research/lm-head-transpose-cache.md` —
  next-tier perf axis identified during this investigation.
- Companion wins (chronological):
  - [`2026-05-19-bench-cpu-matmul-mixed-dispatch-qwen3-06b.md`](2026-05-19-bench-cpu-matmul-mixed-dispatch-qwen3-06b.md) — `lm_head` at 54 % of step (the framing that motivated this tranche)
  - **this entry** — `forward_last_logits` landed, pending production verification

## Codex hand-off ticket

| Field | Value |
|---|---|
| Verification | Re-run `rollout_last_logits_ab_bench` at Qwen3-0.6B-vocab shape, in **exclusive** memory budget (Claude idle) |
| Acceptance | Mean speedup ≥ 1.05× **or** explicit kill if ≤ 1.0× with sigma_pct ≤ 2 % |
| If KILL | Revert this tranche; pivot to lm_head-transpose-cache axis (see research doc) |
| If PASS | Update this entry's `pending-bench` → `verified`, fill production-shape table, cross-link |
