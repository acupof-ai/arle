# Metal MTP Adaptive Fallback

## Context

The short MTP parity smoke exposed the pathological case clearly: token parity
could pass while MTP suffix acceptance stayed at `0.0`. In that shape, MTP pays
draft plus verify cost and saves no target rows.

The user asked for a lower-level fix that can stay stable, including automatic
downgrade after several wrong predictions. This tranche adds a per-request
guard before attempting larger packed-verify or ngram work.

## What Worked

- Added per-request MTP adaptive state:
  `consecutive_zero_accepts`, `fallback_steps_remaining`,
  `adaptive_disable_events`, and `adaptive_fallback_steps`.
- Default policy: after 4 consecutive zero-suffix accepts
  (`accepted_inputs == 1`), pause MTP for 16 output steps.
- During cooldown, the request runs standard target decode via
  `run_standard_decode_with_mtp_seed_capture`, so output correctness stays
  target-owned and MTP can resume with fresh seed hidden.
- Added env controls:
  `ARLE_METAL_MTP_ADAPTIVE=0`,
  `ARLE_METAL_MTP_ZERO_ACCEPT_LIMIT`,
  `ARLE_METAL_MTP_COOLDOWN_TOKENS`.
- Extended `metal_bench` and `--mtp-parity` JSON/human output with adaptive
  disable and fallback counters.

## Verification

- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features metal --bin metal_bench`
- `cargo test -p infer --no-default-features --features metal --bin metal_bench -- --nocapture`
- `cargo check -p infer --no-default-features --features no-cuda`
- `cargo build --release -p infer --no-default-features --features metal --bin metal_bench`

Local parity smoke:

```text
./target/release/metal_bench \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --mtp-draft-model mlx-community/Qwen3.6-35B-A3B-MTP-4bit \
  --mtp-parity \
  --prompt "Write a compact Rust function that reverses a string and explain it briefly." \
  --generation-tokens 8 --json
```

Result: pending rerun after replacing token-count synthetic prompts with real
prompt text. Earlier synthetic-input numbers were implementation smoke only and
are not retained as benchmark evidence.

This smoke validates the downgrade mechanism and parity on one pathological
prompt. It is not a performance claim.

Local MTP-only smoke:

```text
./target/release/metal_bench \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --use-step-driver \
  --mtp-draft-model mlx-community/Qwen3.6-35B-A3B-MTP-4bit \
  --prompt "Write a compact Rust function that reverses a string and explain it briefly." \
  --generation-tokens 16 --warmup 0 --runs 1 --json
```

Result: pending rerun on real prompt text.

## Rule

MTP must stop paying speculative cost indefinitely when recent suffix acceptance
is zero. Adaptive fallback is a stability guard; packed verify and ngram are
separate follow-up optimizations with their own evidence gates.
