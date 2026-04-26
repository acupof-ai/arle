# Metal Qwen3.5 dense fixture: `has_qk_gate` heuristic misclassifies all-FullAttention checkpoints

## Context

After `cae7e05 feat(qwen35): preserve packed gguf weight quantization`, both
Metal CI and the macOS branch of the regular CI started failing on
`tests/cli_tiny_fixture_live.rs::train_test_fixture_checkpoint_runs_through_real_backend_cli`
with:

```
Metal prefill chunk failed for RequestId(0): MLX error:
[reshape] Cannot reshape array of size 9728 into shape (1,152,2,16).
```

The fixture is built by `arle train test`, which routes through the default
`Qwen35Family` (`crates/train/src/bin/pretrain.rs`). With
`linear_attn_every == 0` the attention pattern is all `FullAttention`, so the
saved safetensors carries Qwen3.5-style gated Q (`q_dim = nh*hd*2 = 64`) but
zero GDR/linear layers, and `qwen35_checkpoint.rs` writes
`architectures: ["Qwen2ForCausalLM"]`. The Rust arch detector promotes that
to `ModelArch::Qwen35` because `text_config` is present
(`infer/src/model_registry.rs:204-211`), so the Metal backend loads it via
`Qwen35MetalWeights`.

## Root Cause

`crates/mlx-sys/src/mlx_qwen35_model.cpp::Qwen35CompiledModel::prepare_forward`
auto-detected the Q gate from the layer mix:

```cpp
bool has_gate = (n_gdr > 0);
for (auto& lw : layers) {
    if (!lw.is_gdr) {
        lw.full.has_qk_gate = has_gate;
    }
}
```

That heuristic was added in `a3bf04a` (2026-04-10) for the case where Qwen3
shares the Qwen3.5 C++ struct via `build_qwen3_cpp_model`. It was correct as
long as **only** Qwen3.5 checkpoints with at least one GDR layer reached the
compiled prefill path: pre-`cae7e05`, the `Qwen35MetalWeights::build` route
called `extract_qw`, which returned `None` for `Dense` q/k/v projections and
fell back to the Rust/MLX `qwen35_full_attention_step` (which already
handles the gate at `infer/src/backend/metal/qwen35.rs:2353`).

`cae7e05` introduced `qwen35_compiled_add_dense_weight` to preserve packed
GGUF weights through the C++ path, which also made dense Q/K/V eligible.
With `n_gdr == 0` for the dense fixture, `has_qk_gate` collapsed to `false`,
so the C++ `full_attn_step` reshaped the gated Q output (`(1, S, nh, hd*2)`)
through the non-gate branch (`reshape(q_proj_out, {B, S, nh, hd})`). The
mismatch surfaces as `9728 → (1, 152, 2, 16)` because here `nh == nkv == 2`,
so the failure points at the K reshape line even though the offending
tensor is Q.

## Fix

Make the gate explicit instead of inferred:

- `mlx_qwen35_model.cpp` — replace the `n_gdr > 0` heuristic in
  `prepare_forward` with a per-model flag `model_has_qk_gate` (default
  `false`), and add `qwen35_compiled_set_qk_gate(model, enabled)` to set it.
- `crates/mlx-sys/src/lib.rs` — declare the new FFI.
- `infer/src/backend/metal/qwen35.rs::Qwen35MetalWeights::build` — call
  `set_qk_gate(model, 1)` right after `set_config` (Qwen3.5 always gates Q).
- `infer/src/backend/metal/weights.rs::build_qwen3_cpp_model` — call
  `set_qk_gate(model, 0)` (Qwen3 never gates Q).

Every C++-compiled Qwen3/Qwen3.5 build path now declares the gate state
explicitly, so dense-only Qwen3.5 fixtures and any future hybrid layer mix
are covered.

## Rule

Do not infer per-family invariants from incidental layer counts. When the
Rust builder already knows the family, pass that fact across the FFI as an
explicit flag — heuristics like `n_gdr > 0` only hold for the cases the
author had in mind, and silently misroute the day a different code path
starts feeding the same C++ struct.
