# DSv4 V4 Truth Alignment

## Goal

Phase 0.5 aligned the infer-side DeepSeek scaffold with the actual local
`DeepseekV4ForCausalLM` checkpoint at `infer/models/dsv4-mini-1B-init/`.

This phase is a truth/alignment gate only. It does not license prefill,
decode, MoE, or attention kernels; those remain Phase 1/2 work.

## Hypothesis

If infer uses `DeepSeekV4Config` and validates V4 safetensors names before GPU
allocation, the 2.0 GB init checkpoint should pass config + manifest checks
without depending on CUDA kernels.

Formula-predict: no runtime speed delta, no GPU allocation, no logits yet.
Expected evidence is compile + CPU manifest validation.

## Params

- Model: `infer/models/dsv4-mini-1B-init/`
- Architecture: `DeepseekV4ForCausalLM`
- Checkpoint dtype: `bfloat16`
- Target fields checked: `hidden_size=1024`, `num_hidden_layers=24`,
  `num_key_value_heads=1`, `q_lora_rank=384`, `o_lora_rank=384`,
  `n_routed_experts=16`, `n_shared_experts=1`, `num_experts_per_tok=2`,
  `scoring_func=sqrtsoftplus`, `topk_method=noaux_tc`,
  `num_hash_layers=2`, `sliding_window=64`,
  `num_nextn_predict_layers=1`, `vocab_size=129280`

## Env

- Host: local CUDA workstation
- GPU target: RTX 4070 Ti SUPER / `sm_89`
- Rust profile for executable validation: release
- CUDA kernel build status: blocked by pre-existing nvcc/GCC16 issue in
  `csrc/attention/decode_attention_quantized.cu`

## Results

Before/after:

| Surface | Before | After |
|---|---|---|
| Runtime config | V3-era `DeepSeekConfig` wrapper | `DeepSeekV4Config` wrapper |
| Checkpoint test | `dsv4_nano_smoke` using old nano fixture | `dsv4_v4_1b_smoke` targeting the 2.0 GB V4 checkpoint |
| Tensor truth | MLA/dense names in DeepSeek runtime comments and stubs | V4 names: `wq_a`, `wkv`, hyper-connection, routed experts, shared expert, MTP |
| Loader behavior | DeepSeek V4 not accepted by CUDA bootstrap | CUDA bootstrap detects `DeepSeekV4`, validates manifest, then returns Phase 2A pending error before GPU allocation |
| Manifest gate | none | CPU-only safetensors header validation under `infer/src/deepseek_v4_manifest.rs` |

Verification:

| Command | Result |
|---|---|
| `cargo fmt --all --check` | PASS |
| `git diff --check` | PASS on scoped Phase 0.5 paths |
| `cargo check -p infer --no-default-features --features no-cuda` | PASS |
| `cargo check -p infer --no-default-features --features cuda,no-cuda` | PASS |
| `cargo check -p infer --tests --no-default-features --features cuda,no-cuda` | PASS |
| `cargo test --release -p infer --no-default-features --features no-cuda --lib` | PASS: 569 passed, 11 ignored |
| `cargo check --release -p infer --features cuda` | FAIL: pre-existing nvcc/GCC16 blocker in `decode_attention_quantized.cu` |
| `cargo clippy --release -p infer --features cuda -- -D warnings` | FAIL: same pre-existing nvcc/GCC16 blocker |

The release no-cuda test run includes:

- `deepseek_v4_manifest::tests::v4_config_fields_match_init_checkpoint`
- `deepseek_v4_manifest::tests::v4_tensor_names_fully_covered`
- `deepseek_v4_manifest::tests::v4_checkpoint_manifest_contains_required_tensors`

## Problems

The CUDA release and clippy gates cannot reach Rust typechecking because nvcc
fails while compiling `csrc/attention/decode_attention_quantized.cu` against
`/usr/include/c++/16.1.1/type_traits` (`char8_t` / `requires` errors). This is
the pre-existing blocker called out in the mission brief.

`dsv4_v4_1b_smoke` remains ignored for prefill/decode because V4 attention and
MoE kernels are intentionally not part of Phase 0.5.

## Learnings

The V4 manifest check must stay CPU-only. If it depends on CUDA cfg or CUDA FFI
linking, the project loses the ability to validate checkpoint truth when the
kernel toolchain is broken. Phase 0.5 therefore keeps config/header validation
in an always-available cold-path module and lets the CUDA loader call into it.

Verdict: LICENSE Phase 0.5 truth alignment. Continue with Phase 1 MoE primitive
and Phase 2A V4 forward kernels; do not treat this as a runtime-forward license.
