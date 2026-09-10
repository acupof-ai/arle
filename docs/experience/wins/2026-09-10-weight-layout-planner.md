# Weight-layout planner: one site for the format + shape + SM-tier decision

2026-09-10 · architecture refactor step 5 (weight axis) · no baseline (refactor, no perf claim)

## Context

The weight axis turns a checkpoint into a resident layout: quant-format ABI
(`infer-quant`, landed earlier) and then the repack decision — which kernel
layout a weight gets before it becomes resident. Before this change the
decision was re-derived at six sites: `marlin_repack_dense`, the three
`repack_for_marlin_*` fns in `cuda-kernels`, the W8A16 row-fusion site, and the
DSv4 `decode_proj_cache`. Each carried its own copy of the same
format + shape + compute-capability gates, so a gate fix had to be applied in
N places, and the "one resident layout per weight" invariant
([2026-08-20](2026-08-20-marlin-source-freed-18gb.md)) was assertable nowhere.

The entry audit asked three questions: where do repack decisions live, are
their inputs pure host, and does any decision depend on a device return value.

## What worked

- `infer_quant::plan_weight_layout(query, caps, policy) -> Result<WeightLayoutPlan>`
  is a pure host fn: format, shape, and static device properties in, one plan
  out. Execution stays in `infer-cuda` / `cuda-kernels`. `cargo tree -p
  infer-quant` contains no `cuda-kernels`.
- The three-state semantics live on the types, with the why as a one-line doc:
  `RepackRequirement::Required` exists because NVFP4's one serving arm is
  Marlin (a device that cannot build it has nowhere to go, so the load fails);
  W8A16 and per-channel FP8 have a scalar/dequant fallback, so their repack is
  `Optional` and a failed gate demotes with one warning.
- The audit's one device-return-value input — the DeepGEMM native preflight,
  which looks like an env check but is a cached device probe — enters the
  planner as a host bool (`LayoutPolicy.deepgemm_native_available`) filled at
  the call site. The planner itself touches no device.
- The exit gate was checked by grep, not by `cargo tree`: the
  format + cc + shape structure appears exactly once in the tree
  (`layout.rs`); the three executor repack fns now carry only format +
  source-buffer-presence guards. `cargo tree` proves only the dependency
  boundary; the grep proves the boundary earned its keep.
- `prepare_fp4_deepgemm_sfb` carried a second, silent copy of the sfb shape
  gate (identical to the planner's). A planner/executor disagreement would
  have skipped the arm with no signal — failure and success looking identical.
  The silent no-op is now a hard bail.
- Two availability predicates (`fp4_deepgemm_available`,
  `fp8_deepgemm_per_channel_available`) and the `Fp4Query.prefill_shape` field
  were deleted: the sfb buffer's presence is the plan's decision, so the route
  condition collapses to `sfb && m >= floor`.
- Tests: 8 planner tests including the negative controls (NVFP4 on sm_75 is a
  hard error; group_size 33 and K=48/gs=32 demote). Mac CUDA clippy gate green.
  11 files changed, +585/−176.

## Rule

- A decision structure (a gate's conditions combined) appears once in the
  tree; executors keep only state guards. Whether an extraction earned its
  boundary is proved by grepping for the structure, not by the dependency
  graph.
- When two components carry the same gate, their disagreement must be loud
  (assert/bail). A silent no-op that duplicates a decision is a place where
  failures become indistinguishable from normal operation.
- A device probe result crosses into a host planner as a host value, filled by
  the caller that is allowed to touch the device. Both the dependency graph
  and the decision's purity survive.
