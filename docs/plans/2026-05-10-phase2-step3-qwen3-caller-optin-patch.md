---
title: M_rope-yarn-scaling Phase 2 step 3 — qwen3/weights.rs caller opt-in patch design
date: 2026-05-10
type: plan
status: ready-to-apply-post-codex
audience: Claude (next tick) OR codex pickup
---

# Phase 2 step 3 — qwen3/weights.rs precompute_rope opt-in patch

> Pre-built ~10-LOC patch design for qwen3/weights.rs caller opt-in to
> `precompute_rope_with_scaling`. Apply after codex commits #24 W4A8
> prefill graph capture hoist(currently WIP on this file)to avoid
> race-condition contamination.

## Why deferred

`infer/src/model/qwen3/weights.rs` 当前 codex WIP for #24:
- Line 269:`rope_scaling: None` added to Qwen3Config gguf path constructor(他 anticipated 我 field add)
- Line 672-678:`marlin_prefill_scratch_config()` 加入(#24-specific MarlinPrefillScratch lifecycle)
- Line 858:`rope_scaling: None` test fixture

Edit + `git add` 会污染 codex 的 WIP commit。等他 commit 后立即 apply。

## Patch design(等 commit 后 1-tick apply)

### Hunk 1 — import line 14-18

```rust
- use crate::weight_loader::{
-     QuantLoadConfig, load_tensor_1d, load_tensor_2d, load_tensor_2d_concat_rows,
-     load_tensor_2d_maybe_quantized_with_config, load_tensor_2d_sharded, precompute_rope,
-     resolve_rope_cache_len,
- };
+ use crate::weight_loader::{
+     QuantLoadConfig, load_tensor_1d, load_tensor_2d, load_tensor_2d_concat_rows,
+     load_tensor_2d_maybe_quantized_with_config, load_tensor_2d_sharded,
+     precompute_rope_with_scaling, resolve_rope_cache_len,
+ };
```

### Hunk 2 — safetensors loader call site line 447-449

```rust
- let (cos_cache, sin_cache) =
-     precompute_rope(&ctx, config.head_dim, rope_cache_len, config.rope_theta)?;
+ let (cos_cache, sin_cache) = precompute_rope_with_scaling(
+     &ctx,
+     config.head_dim,
+     rope_cache_len,
+     config.rope_theta,
+     config.rope_scaling.as_ref(),
+ )?;
```

### Hunk 3 — GGUF loader inner-fn import line 691-693

```rust
- use crate::weight_loader::{
-     load_tensor_1d_gguf, load_tensor_2d_gguf, load_tensor_2d_gguf_bf16, precompute_rope,
- };
+ use crate::weight_loader::{
+     load_tensor_1d_gguf, load_tensor_2d_gguf, load_tensor_2d_gguf_bf16,
+     precompute_rope_with_scaling,
+ };
```

### Hunk 4 — GGUF loader call site line 748-750

```rust
- let (cos_cache, sin_cache) =
-     precompute_rope(ctx, config.head_dim, rope_cache_len, config.rope_theta)?;
+ let (cos_cache, sin_cache) = precompute_rope_with_scaling(
+     ctx,
+     config.head_dim,
+     rope_cache_len,
+     config.rope_theta,
+     config.rope_scaling.as_ref(),
+ )?;
```

## Type compatibility

`Qwen3Config::rope_scaling: Option<qwen3_spec::RopeScalingConfig>` matches
`precompute_rope_with_scaling`'s `Option<&qwen3_spec::RopeScalingConfig>`
parameter directly via `.as_ref()` — **no conversion shim needed**(unlike
qwen35-spec which required `qwen35_to_qwen3_rope_scaling`).

## Verification

post-apply:
```bash
cargo check -p infer --no-default-features --features no-cuda
cargo test -p qwen3-spec --lib  # vanilla noop test still passes (existing)
```

vanilla path bit-equivalent because `Qwen3Config::rope_scaling` defaults to
`None` for all current Qwen3-4B model configs(serde default)。No behavior
change for existing workloads。

## ROI summary

LOC: ~+10 / -4(2 import lines + 2 call site rewrites)
Risk: 0(pure additive opt-in,vanilla path unchanged)
Wall-clock:5 min apply + 2 min cargo check post codex commit

## Cross-references

- Phase 2 step 2 precedent:`cb80829`(qwen35-spec mirror impl pattern)
- Phase 2 step 1 wrapper:`d5f67b4`(precompute_rope_with_scaling 加 weight_loader.rs)
- Codex WIP scope:#24 W4A8 prefill graph capture hoist(`docs/plans/2026-05-09-prefill-graph-phase0v3-validation-protocol.md`)
- M_rope-yarn-scaling consolidated:`docs/experience/wins/2026-05-10-m-rope-yarn-scaling-phase1-phase2-landed.md`

## 状态

Phase 2 step 3 patch ready-to-apply,等 codex #24 commit 释放
qwen3/weights.rs。Apply 后 M_rope-yarn-scaling Phase 2 全闭合(只剩 Phase
2 step 4 Metal sync 需 Mac + Phase 3 long-ctx bench validation)。
