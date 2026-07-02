# 2026-05-05 · DeepSeek V4 registry routing

## Status: local verification complete; benchmark not performance-applicable

This tranche lands DS1 model-registry routing for DeepSeek V4 and V4 MTP. It is
CPU-only config dispatch: no scheduler loop, CUDA kernel, KV layout, loader, or
HTTP serving path changes. The runtime continues to reject every DeepSeek family
through `ModelArch::is_implemented=false` / CUDA bootstrap unsupported routing
until the DS3/DS4/DS5 serving work lands.

## Goal

Make V4 configs fail explicitly as known-but-unimplemented ARLE model families
instead of falling into the unknown-architecture path.

## Hypothesis

Registry-only dispatch has no TTFT/ITL/tok-s effect because it runs once during
model bootstrap and does not change loaded Qwen runtime paths.

## Params

- Affected code: `infer/src/model_registry.rs`.
- New detected architectures: `DeepseekV4ForCausalLM`, `DeepseekV4MTP`.
- Runtime status: DeepSeek V4 and V4 MTP remain unimplemented.

## Env

- Host: local CI/container workspace.
- GPU: not required.
- Feature set: default workspace checks plus no-cuda check.

## Results

| Check | Result |
|---|---:|
| `cargo test -p infer --release model_registry` | passed: 15 registry tests |
| `cargo clippy --workspace --all-targets --exclude infer -- -D warnings` | passed |
| `cargo clippy --workspace --all-targets -- -D warnings` | not green: pre-existing infer pedantic drift remains outside this registry diff |
| guidellm throughput | not run: no serving hot-path delta |

## Problems

No runtime benchmark is meaningful until a DeepSeek serving path exists. The
bench discipline is satisfied here by recording the non-applicability instead
of silently skipping the entry for an `infer/src/` diff.

## Learnings

Treat V4 as its own attention family in the registry. Reusing
`AttentionVariant::Mla` would hide the V4 hybrid local + long-range sparse
attention work behind the V3 label and invite the wrong kernel design.
