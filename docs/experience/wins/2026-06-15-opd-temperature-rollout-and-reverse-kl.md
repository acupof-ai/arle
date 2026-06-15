# OPD temperature rollout + reverse-KL — smoke + V100 directional A/B, 2026-06-15

## SLO-shape probed? N — V100 directional only

This entry records mechanism bring-up for two opt-in OPD knobs plus a directional
V100 CPU A/B on real Qwen3.5-0.8B. This is not a capability run; the full-step
multi-seed capability gate stays reserved for the H20 OPD-after-QAT phase. No
default flip is licensed by this entry.

## Roofline check

N/A. This tranche changes OPD rollout/loss selection, not a runtime kernel or
serving hot path. Performance and quality attribution are deferred to the
matched V100 runs below.

| Op | Achieved | Peak (this HW) | % | Verdict |
|---|---:|---:|---:|---|
| OPD tiny CPU smoke | n/a | n/a | n/a | deferred: real A/B pending |

## Goal

Expose two unresolved OPD levers without perturbing defaults: temperature
sampling for rollout support, and reverse KL for the steady-state distillation
objective. Quantify the V100 s/step cost because the OPD budget is steps/hour
bound.

## Hypothesis

`--rollout-temperature > 0` should lift the cold-start CE anchor above the
near-zero greedy self-argmax loss, while `--kl-direction reverse` should be a
safe opt-in scaffold for a matched forward-vs-reverse quality A/B.

## Command

Local verification used the CPU/no-cuda training smoke path, not guidellm:

```bash
cargo test -p train -p cli -p infer-plan --release --no-default-features --features cpu,no-cuda
cargo clippy -- -D warnings
CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
cargo build --release --no-default-features --features cpu,no-cuda,cli
target/release/arle train self-opd --smoke --steps 5
target/release/arle train self-opd --smoke --steps 5 --rollout-temperature 1.0 --rollout-seed 42
target/release/arle train self-opd --smoke --steps 5 --kl-direction reverse
```

## Environment

- **Backend:** CPU autograd (`--features cpu,no-cuda,cli`).
- **Model:** embedded tiny Qwen3.5 config for smoke; V100 A/B used
  `/home/chenkailun.c/.cache/modelscope/hub/models/Qwen/Qwen3.5-0.8B-Base`.
- **Hardware:** local Mac CPU for smoke; CUDA typecheck via `cuda,no-cuda`.
- **V100 run dir:** `/home/chenkailun.c/opd-v100-verify-5b08140f-20260615-233320`.
- **Commits:** reverse-KL routing fix `3b9e311e`; profile surface `037eef89`;
  pure-OPD eval surface `5b08140f`; real `train opd` LoRA student fix
  `9315b63a`.
- **Feature set:** `cargo build --release --no-default-features --features cpu,no-cuda,cli`.
- **Non-default flags / env vars:** `--rollout-temperature 1.0 --rollout-seed 42`,
  `--kl-direction reverse`, `--gkd-lambda 0.0`, `ARLE_OPD_STEP_PROFILE=1`;
  `CUDARC_CUDA_VERSION=12080` for CUDA-Rust typecheck.
- **Server launch:** n/a.

## Results — smoke evidence

### Baseline identity

Default greedy/forward smoke stayed byte-identical to explicit
`--kl-direction forward` (`cmp` exit 0) and matches the pre-P0 trace.

| Mode | Loss trace |
|---|---|
| default greedy/forward | `0.164029 -> 0.164008 -> 0.163986 -> 0.163964 -> 0.163940` |
| explicit `--kl-direction forward` | byte-identical to default |

### P0 sampling

| Mode | Loss trace | Verdict |
|---|---|---|
| greedy default | `0.164029 -> 0.164008 -> 0.163986 -> 0.163964 -> 0.163940` | unchanged |
| `--rollout-temperature 1.0 --rollout-seed 42` | `0.172304 -> 0.172276 -> 0.171992 -> 0.172070 -> 0.172055` | differs from greedy |
| same seed 42 rerun | same as seed 42 | deterministic (`cmp` exit 0) |
| seed 43 | `0.172685 -> 0.172666 -> 0.172647 -> 0.172628 -> 0.172608` | differs from seed 42 |

`target/release/arle train opd --smoke --steps 5 --rollout-temperature 1.0 --rollout-seed 42`
also exited 0.

### P1 reverse KL

| Command | Result |
|---|---|
| `target/release/arle train self-opd --smoke --steps 5 --kl-direction reverse` | exit 0, trace `0.077914 -> 0.077893 -> 0.077871 -> 0.077848 -> 0.077825` |
| `target/release/arle train opd --smoke --steps 5 --kl-direction reverse` | exit 0 |

New reverse-KL unit tests:

| Test | Result |
|---|---|
| `loss::tests::reverse_kl_matches_hand_computed_two_row_reference` | PASS |
| `loss::tests::reverse_kl_identical_logits_is_zero` | PASS |
| `loss::tests::reverse_kl_random_finite_logits_is_non_negative` | PASS |

## Results — verification gates

| Gate | Result |
|---|---|
| `cargo test -p train -p cli -p infer-plan --release --no-default-features --features cpu,no-cuda` | PASS (`cli` 153, `infer-plan` 11, `train` 93, train integration/doc tests PASS) |
| `cargo clippy -- -D warnings` | PASS |
| `CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib` | PASS |
| `git diff --check` | PASS |

## Results — V100 directional A/B

Prompt IDs:
`760,6511,314,9338,369,11751,11,321,279,242476,300,21262,12965,303,1141,3990,13`.
Held-out IDs:
`27336,85895,506,799,7493,11995,59275,506,9117,2119,13`.

Per the speed-call update, the already-near-done greedy arm was allowed to finish
50 steps; the fair P0 comparison uses its first 20 steps. Remaining arms ran
20 steps with `--gate-every-n 5`.

### P0 — greedy vs sampled rollout

| Arm | Held-out NLL trajectory | Final vs baseline | Mean s/step | Phase means: rollout / teacher / student / KL / backward | s/step delta |
|---|---|---:|---:|---|---:|
| greedy first 20 | `2.371392 -> 2.363729 -> 2.356122 -> 2.348320 -> 2.339962` | `-0.031430` | `140.750` | `105.724 / 7.634 / 7.677 / 2.351 / 17.357` | baseline |
| `--rollout-temperature 1.0` | `2.371392 -> 2.366468 -> 2.360371 -> 2.355478 -> 2.350288` | `-0.021104` | `141.756` | `106.619 / 7.595 / 7.743 / 2.345 / 17.446` | `+0.714%` |

Training-loss lift was too small to interpret quality: greedy mean loss
`7.26e-6`, sampled mean loss `8.35e-6`. This is a barely-learning cold-start run,
so the sampled-vs-greedy `+0.010326` NLL gap at step 20 is noise from a
confounded setup, not a sampling-method verdict. The valid P0 fact from this
run is performance only: temperature sampling added `+0.714%` s/step.

Greedy full 50 for reference:
`2.371392 -> 2.363729 -> 2.356122 -> 2.348320 -> 2.339962 -> 2.330731 -> 2.320419 -> 2.308982 -> 2.296615 -> 2.283705 -> 2.270887`;
mean `140.908 s/step`.

### P1 — forward vs reverse pure KL

Final pure `train opd` uses a LoRA student (`rank=16`, `alpha=32`,
`attention-qv`) because a full-trainable real-checkpoint probe reached baseline
NLL then was killed with `rc=137` after 216s before step 1 completed on the 31
GiB V100 CPU host.

| Arm | Held-out NLL trajectory | Final vs baseline | Mean s/step | Phase means: rollout / teacher / student / KL / backward | s/step delta |
|---|---|---:|---:|---|---:|
| forward KL | `2.371392 -> 2.371392 -> 2.371391 -> 2.371392 -> 2.371395` | `+0.000003` | `142.759` | `107.476 / 8.039 / 7.676 / 10.201 / 17.032` | baseline |
| reverse KL | `2.371392 -> 2.371391 -> 2.371389 -> 2.371389 -> 2.371387` | `-0.000005` | `140.195` | `104.973 / 7.838 / 7.607 / 10.364 / 17.010` | `-1.796%` |

Reverse is now proven reachable on the fixed pure-KL `train opd` path. The
forward-vs-reverse NLL read is confounded: teacher and student were the same
checkpoint, so KL was approximately zero and the run had no meaningful
teacher>student gradient. The `-0.000008` step-20 NLL delta is therefore a setup
artifact, not evidence that reverse KL is flat, good, bad, or merely
opt-in-only.

## Problems

- Local smoke is mechanism-only; it does not prove capability or default-flip
  safety.
- P0/P1 public config/signature changes required mechanical updates in direct
  loss tests and profile/example harnesses so old forward behavior still
  compiles and stays explicit.
- The first real `train opd` attempt exposed two independent gaps: the student
  was loaded frozen, then full-trainable real-checkpoint OPD was killed on V100
  (`rc=137`). The shipped real path uses a LoRA student, matching the viable
  self-OPD memory profile.
- The V100 P0/P1 A/Bs are confounded directional reads, not capability verdicts:
  P0 stayed near the cold-start `~8e-6` loss floor, and P1 used
  teacher==student. A validated method failing in this setup would be our setup
  failing, not the method.

## Learnings

- Branch on `None` for default rollout sampling. Treating temperature 0 as a
  sampled argmax would not prove byte identity because it would route defaults
  through a new primitive.
- Reverse KL must keep the `q * log q` entropy term in the autograd graph; it is
  not constant because the student distribution is trainable.

## Delta Vs Baseline

- **Baseline:** [`2026-06-14-sopd-91-self-opd-subcommand-inline-loop.md`](2026-06-14-sopd-91-self-opd-subcommand-inline-loop.md)
- **Delta:** defaults unchanged in smoke. V100 P0 sampling was `+0.714%`
  s/step; the NLL delta is not interpretable because the run barely learned.
  Reverse KL was reachable after the fix and measured `-1.796%` s/step vs
  forward; the NLL delta is not interpretable because teacher==student makes
  KL approximately zero.

## Artefacts

- Local smoke captures: `/tmp/arle-p1-final-self-opd-default.txt`,
  `/tmp/arle-p1-final-self-opd-sampled-seed42.txt`,
  `/tmp/arle-p1-final-self-opd-sampled-seed43.txt`,
  `/tmp/arle-p1-final-self-opd-reverse.txt`.
- V100 logs:
  `/home/chenkailun.c/opd-v100-verify-5b08140f-20260615-233320/logs/ab1_greedy.log`,
  `ab1_sampled20.log`, `ab2_forward20_lora.log`, `ab2_reverse20_lora.log`,
  `ab2_forward_probe2_fixed.log`.

## Notes

- What changed in code: P0 added opt-in rollout sampling; P1 added opt-in
  reverse KL; fixes `3b9e311e`, `4879d598`, and `9315b63a` make reverse/pure-KL
  real runs reachable.
- Follow-ups: run the real capability test with a teacher>student gap
  (4B -> 0.8B) on GPU during OPD-after-QAT; V100 results here are directional
  setup/perf reads only.
