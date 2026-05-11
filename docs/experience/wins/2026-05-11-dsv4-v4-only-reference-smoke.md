# DSv4 V4-Only Reference Smoke

## Goal

Delete old DeepSeek V2/V3/nano compatibility paths and make the local
`infer/models/dsv4-mini-1B-init` checkpoint run through real V4 inference and
train bootstrap flows.

## Hypothesis

If the spec, registry, train command, and CPU smoke backend all use
`DeepSeekV4Config` plus the V4 tensor-name contract, then the 1B init checkpoint
can be loaded and executed without retaining any V3/MLA/nano truth surface.

## Params

- Model: `infer/models/dsv4-mini-1B-init`
- Architecture: `DeepseekV4ForCausalLM`
- Checkpoint: single `model.safetensors`, 1889 tensors required / present
- Inference path: `cpu_serve` slow Rust reference, greedy, prompt `"!"`,
  `max_tokens=1`
- Train path: `arle train pretrain-dsv4 --corpus crates/train/data/sample.txt
  --out /tmp/arle-dsv4-train-smoke --steps 0 --batch 1 --seq 8`

## Env

- Host: local dev machine, CPU reference path
- Build: `--release --no-default-features --features cpu,no-cuda`
- CUDA kernels: not used for the reference smoke

## Results

| Check | Result |
| --- | --- |
| `cargo test --release -p infer --no-default-features --features cpu,no-cuda dsv4_reference_one_token_forward_logits_shape -- --ignored --nocapture` | PASS, full 24-layer one-token forward, logits len `129280`, finite |
| `cpu_serve` HTTP `/v1/completions` prompt `"!"`, `max_tokens=1` | PASS, `prompt_tokens=1`, `completion_tokens=1`, text `"进来了"` |
| `arle train pretrain-dsv4 ... --steps 0 --batch 1 --seq 8` | PASS, `tokens=1053`, `tensors=1889/1889`, writes seeded V4 checkpoint |
| `cargo check -p agent-infer --no-default-features --features cpu,no-cuda` | PASS |
| `cargo clippy -p infer --no-default-features --features cpu,no-cuda -- -D warnings` | PASS |
| `cargo clippy -p agent-infer --no-default-features --features cpu,no-cuda -- -D warnings` | PASS |
| `cargo test -p deepseek-spec` | PASS, 3 tests |
| `cargo test -p train --no-default-features pretrain_dsv4 -- --nocapture` | PASS, 4 tests |

## Problems

- This is a correctness/reference smoke, not a serving benchmark. CUDA V4
  hybrid attention, MoE grouped GEMM, MTP, and paged decode kernels are still
  pending.
- The first real forward run killed an over-strict spec assumption: hash-routed
  `tid2eid` can select the same expert more than once. The helper now validates
  range only and preserves duplicate accumulation semantics.
- `pretrain-dsv4` now seeds from the V4 1B init checkpoint and writes a train
  manifest; optimizer update/backward for full V4 remains pending.

## Learnings

The V4 truth surface must be tested by executing the actual 1B tensors. Manifest
coverage alone missed the duplicate hash-route case; a one-token forward caught
it immediately.
