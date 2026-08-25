# Global dead-code deletion, wave 5 — CLI flags + dead pub surface, 2026-08-25

> Status: verified (local Mac: workspace clippy cpu lane + CUDA lint lane + Metal lane)

## Goal

Fifth wave. New slice vs waves 1–4: trace every CLI flag in `crates/cli/src/args.rs`
end-to-end (definition → mapping → consumption site), plus dead pub surface in
infer-cuda / infer-core / infer-server / infer-api and the train-spec crates.
Zero runtime change is the correctness criterion.

## Scope

31-agent workflow (5 slice finders + one adversarial verifier per candidate):
26 candidates, **25 confirmed dead and deleted, 1 rejected** (below).

| Area | Deleted |
|------|---------|
| cli | `--json` flags on `train w2s` / `train rubric-opd` / `train agent-opd` — parsed but no driver ever read `args.json` (sibling `--json` on opd/self-opd/ppl/env is consumed) |
| infer-cuda | `CapturedDecodeGraph` (speculative per-bucket wrapper), `CudaGraphState::rearm_warm` (zero callers since slot-reset switched to graph rebuild), `TpRuntime::single` / `TpRuntime::comm`, `reset/print_dsv4_linear_profile` wrappers (the underlying `linear_profile::reset`/`print_rank0` stay — `dsv4/decode_batch.rs` calls them directly; the stage pair survives via `examples/dsv4_parity.rs`) |
| infer-core | `SchedulerConfig.prefill_max_requests` (never set to `Some`; wrapper collapses to `running_cap()`), `SchedulerConfig.prefix_cache_low_water_pages` + `evict_prefix_cache_if_below_low_water` (always 0 → permanent no-op; live path is `evict_prefix_cache_for_pages`), `Engine::new` (all sites use `with_config`), `pub use radix::{BlockId, PrefixMatch, RadixCache}` (internal uses repointed to `crate::radix::`) |
| infer-server / infer-api | Dead re-exports `coordinator_router`, `bind_and_serve` (functions live via internal paths) |
| autograd | `all_gather_seq` op cluster (op + backward + `BackwardOp` variant + `reduce_scatter_sum_device` trait method + CUDA impl — dead since CP switched to ring attention 2026-07-30); test-only `exp` chain (op + backward + `Backend::exp`/`exp_forward`/`exp_backward_device` + Metal override + `UnaryOp::Exp` + CUDA overrides + `exp_f32`/`exp_backward_f32` kernel registry entries + `cpu_exp_forward` + 3 parity tests) |
| qwen35-spec | `compute_attention_factor` (YARN fields parsed and carried but never applied), `validate_train_dense_full_attention_contract` + its dead `dense_qwen35_config` test helper |
| misc | `Qwen35TensorKind::is_norm`, `TpConfig::from_env`/`from_lookup`, `ShardingSpec::is_full`, `TopoError::message` (redundant with `Display`), `BufferedDiffusionExecutor::into_inner`, `bundle_name` shell fn |

## Rejected (do not re-litigate)

`Qwen35Model::new_with_lora_targets_and_tp` / `_layer_start` + the 4 train
parity examples that are their only consumers (`a2_qwen35_tp_lora_fd`,
`cp_hidden_parity`, `nd_parallel_parity`, `moe_tp_parity`). They have no CI or
script invocation, but `cp_hidden_parity` was used on 2026-08-16 to bisect the
ring-attention regression (errors/2026-08-19) — the benches are the only
multi-GPU LoRA/CP parity coverage, run manually via their doc headers.
Methods and examples stay together (deleting one without the other is a
half-state).

## Note

`BufferedDiffusionExecutor::into_inner` was listed as deleted in the wave 4
entry but was still in the tree (commit `aa10fccca` removed `new`, missed
`into_inner`). Wave 5 completed it.

The first pass also over-deleted `linear_profile::reset()`: the finder's
"only caller is the dead wrapper" claim missed `dsv4/decode_batch.rs:91`,
which calls it directly. The pre-push hook's clippy lane caught it (the
local CUDA lint lane had masked its exit code through a `| tail` pipe —
a pipeline reports `tail`'s status, not cargo's).

## Rule

A sweep's rejection log is part of the artifact: record why a surviving
candidate stays so the next sweep doesn't re-litigate it.

A lint lane piped through `tail`/`grep` must use `set -o pipefail` (or check
`pipestatus`), or the lane reports green when the compiler failed.
