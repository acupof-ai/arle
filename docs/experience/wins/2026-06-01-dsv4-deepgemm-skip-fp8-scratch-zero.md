# DSv4 DeepGEMM skips FP8 scratch zeroing

## SLO-shape probed? -- pending remote

## Goal

Move the current routed-expert path closer to the SGLang MoE MLP shape by
removing pure buffer materialization work from the DeepGEMM grouped expert
path.

## Hypothesis

`forward_deepgemm_grouped_dsv4_experts_gpu` quantizes only valid rows for each
active expert and passes `masked_m` into the native grouped DeepGEMM bridge.
The later unpad/scatter kernels also read only rows with `row < count`.
Therefore clearing the large FP8 input and activation scratch buffers on every
call is unnecessary. Scale buffers remain zeroed so TMA-aligned scale padding is
deterministic.

## What Changed

- Default path skips `dg.input_fp8` and `dg.act_fp8` full-buffer `memset_zeros`.
- `ARLE_DSV4_DEEPGEMM_ZERO_FP8_SCRATCH=1` restores the old behavior for A/B or
  rollback.
- `dg.input_scales` and `dg.act_scales` are still cleared.

## Verification

Local non-CUDA gates passed before remote CUDA validation:

- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `cargo test -p infer --no-default-features --features no-cuda tensor_parallel -- --nocapture`
- `git diff --check HEAD`

Remote CUDA build, correctness, and performance A/B are pending in the pod
because the change is under `#[cfg(feature = "cuda")]`.

## Rule

For SGLang-path MoE work, prefer masked valid-row contracts over full scratch
clears. Keep a rollback env switch until the exact SLO workload validates both
normal output and wall-clock improvement.
