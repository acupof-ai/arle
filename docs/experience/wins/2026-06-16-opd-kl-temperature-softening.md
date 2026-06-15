# OPD KL-temperature softening scaffold

## Context

P2 adds an opt-in KL softening temperature for pure OPD. This is a mechanism
tranche only: no capability verdict or default flip is claimed locally.

## What Worked

- `kl_distill_loss` and `kl_distill_loss_chunked` now divide both teacher and
  student logits by `T` before softmax/log-softmax, then apply the `T^2`
  distillation compensation once to the assembled KL scalar.
- `T=1.0` is covered by a byte-identity unit test against the untempered KL
  path, including student-logit gradients.
- `GkdLossConfig` rejects `kl_temperature != 1.0` when `lambda > 0.0` because
  the SFT anchor is intentionally `1/vocab`-scale-matched to KL; applying `T^2`
  only to KL would silently reweight `(1-lambda)KL + lambda*SFT`.
- CLI surfaces `--kl-temperature` on `train opd` and `train self-opd`; values
  must be positive. The held-out gate remains per-token NLL, independent of KL
  temperature.

## Verification

Local CPU gates:

```bash
cargo test -p train --release kl_temperature -- --nocapture
cargo test -p train --release chunked_kl_matches_baseline_with_temperature -- --nocapture
cargo test -p train --release validate_gkd_loss_config_rejects_temperature_with_sft_blend -- --nocapture
```

All passed.

## Bench

pending-remote: run the CUDA pod pure-OPD A/B with `--gkd-lambda 0.0` and sweep
`--kl-temperature {1,2,3}`. Verdict metric is T-independent held-out per-token
NLL; report s/step as a secondary cost metric.

## Rule

KL temperature is a pure-OPD lever only until a separate design intentionally
rescales the SFT anchor with the same semantics.
