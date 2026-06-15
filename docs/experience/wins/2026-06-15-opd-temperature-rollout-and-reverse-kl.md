# OPD temperature rollout + reverse-KL scaffold — smoke gate, CPU/no-cuda, 2026-06-15

## SLO-shape probed? N — tiny smoke only

This entry records mechanism bring-up for two opt-in OPD knobs. No SLO workload
or capability run was performed locally; the real Qwen3.5-0.8B A/B stays
`pending-remote` below. No default flip is licensed by this entry.

## Roofline check

N/A. This tranche changes OPD rollout/loss selection, not a runtime kernel or
serving hot path. Performance and quality attribution are deferred to the
matched V100 runs below.

| Op | Achieved | Peak (this HW) | % | Verdict |
|---|---:|---:|---:|---|
| OPD tiny CPU smoke | n/a | n/a | n/a | deferred: real A/B pending |

## Goal

Expose two unresolved SOPD levers without perturbing defaults: temperature
sampling for rollout support, and reverse KL for the steady-state distillation
objective.

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

- **Backend:** CPU autograd smoke (`--features cpu,no-cuda,cli`).
- **Model:** embedded tiny Qwen3.5 config for smoke; real Qwen3.5-0.8B is pending remote.
- **Hardware:** local Mac CPU for smoke; CUDA typecheck via `cuda,no-cuda`.
- **Commits:** P0 `b092f4aa`, P1 `6dddf310`; smoke rerun on current main `68a331a0`.
- **Feature set:** `cargo build --release --no-default-features --features cpu,no-cuda,cli`.
- **Non-default flags / env vars:** `--rollout-temperature 1.0 --rollout-seed 42`,
  `--kl-direction reverse`; `CUDARC_CUDA_VERSION=12080` for CUDA-Rust typecheck.
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

## Pending Remote A/B

Run on the V100 box with real `Qwen/Qwen3.5-0.8B-Base`, CPU autograd, matching
the 2026-06-14 SOPD verification protocol. Token IDs below were encoded from
`models/Qwen3.5-0.8B/tokenizer.json`.

```bash
ssh v100 <<'EOF'
set -euo pipefail
cd ~/code/agent-infer
git fetch origin
git checkout main
git pull --ff-only
cargo build --release --no-default-features --features cpu,no-cuda,cli

MODEL=${MODEL:-/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3.5-0.8B-Base}
PROMPT_IDS=760,6511,314,9338,369,11751,11,321,279,6511,314,9564,369,19241,13
EVAL_IDS=27336,85895,506,799,7493,11995,59275,11,321,90162,506,6942,11995,59275,13
COMMON="--student-model ${MODEL} --backend cpu --steps 20 --rollout-len 16 --prompt-ids ${PROMPT_IDS} --eval-ids ${EVAL_IDS} --gate-every-n 1 --gate-regress-tol 0.02 --json"

target/release/arle train self-opd ${COMMON} \
  > /tmp/sopd-p0-greedy.json
target/release/arle train self-opd ${COMMON} \
  --rollout-temperature 1.0 --rollout-seed 42 \
  > /tmp/sopd-p0-temp1-seed42.json

target/release/arle train self-opd ${COMMON} --kl-direction forward \
  > /tmp/sopd-p1-forward.json
target/release/arle train self-opd ${COMMON} --kl-direction reverse \
  > /tmp/sopd-p1-reverse.json
EOF
```

P0 license: the temperature run must improve held-out NLL by a clear margin over
greedy and lift the training-loss magnitude above the real-prompt ~8e-6 cold
start floor; otherwise kill the lever and keep greedy default.

P1 license: compare forward vs reverse on held-out NLL/capability. A default
flip needs full MMLU multi-seed >=5 with mean +/- sigma and Wilson 95% CI per
the 2026-05-28 rule. Until that gate passes, both knobs remain opt-in.

## Problems

- Local smoke is mechanism-only; it does not prove capability or default-flip
  safety.
- P0/P1 public config/signature changes required mechanical updates in direct
  loss tests and profile/example harnesses so old forward behavior still
  compiles and stays explicit.

## Learnings

- Branch on `None` for default rollout sampling. Treating temperature 0 as a
  sampled argmax would not prove byte identity because it would route defaults
  through a new primitive.
- Reverse KL must keep the `q * log q` entropy term in the autograd graph; it is
  not constant because the student distribution is trainable.

## Delta Vs Baseline

- **Baseline:** [`2026-06-14-sopd-91-self-opd-subcommand-inline-loop.md`](2026-06-14-sopd-91-self-opd-subcommand-inline-loop.md)
- **Delta:** defaults unchanged in smoke; new sampled/reverse paths are opt-in
  and pending real Qwen3.5-0.8B A/B.

## Artefacts

- Local smoke captures: `/tmp/arle-p1-final-self-opd-default.txt`,
  `/tmp/arle-p1-final-self-opd-sampled-seed42.txt`,
  `/tmp/arle-p1-final-self-opd-sampled-seed43.txt`,
  `/tmp/arle-p1-final-self-opd-reverse.txt`.
- Remote outputs: pending `/tmp/sopd-p0-*.json`, `/tmp/sopd-p1-*.json` on V100.

## Notes

- What changed in code: P0 `b092f4aa` adds opt-in rollout sampling; P1
  `6dddf310` adds opt-in reverse KL.
- Suspected cause of any regression: n/a locally; remote A/B may expose quality
  regressions from mode-seeking reverse KL.
- Follow-ups: run the V100 A/B above before making any quality or default-flag
  claim.
