# DSv4 CUDA Real Loader Phase 1.5 Stopped

## Context

Phase 1.5 was intended to turn the Phase 0 CUDA DSv4 zero-logits shell into a
real one-token decode path by reusing the existing Qwen3.5 safetensors loader
and CUDA ops, then making the Phase 1 parity test pass against the CPU
reference oracle.

This entry cites the Phase 1 parity STOP commit:

- `00bdb4d1e4778d9afde2ee289cefbe151a6b5154`

## Root Cause

Two license-or-kill gates fire before implementation.

### Gate 1: no dense FFN fallback

Config evidence from `infer/models/dsv4-mini-1B-init/config.json`:

| Field | Value |
| --- | --- |
| `num_hidden_layers` | `24` |
| `n_routed_experts` | `16` |
| `n_shared_experts` | `1` |
| `num_experts_per_tok` | `2` |
| `num_hash_layers` | `2` |
| attention modes | `3` sliding-window, `11` CSA, `10` HCA |
| `compress_ratios` | `[0,0,4,96,4,96,4,96,4,96,4,96,4,96,4,96,4,96,4,96,4,96,4,0]` |

Spec evidence:

- `crates/deepseek-spec/src/v4.rs:161-168` calls
  `layer_tensor_names(layer_idx)` with `include_shared_experts =
  self.n_shared_experts > 0` for every layer.
- `crates/deepseek-spec/src/v4.rs:762-787` defines every layer FFN as
  `DeepSeekV4MoeTensorNames` with `gate.weight`, optional `gate.bias` or
  `gate.tid2eid`, routed `experts.*`, and optional `shared_experts`.
- `infer/src/deepseek_v4_manifest.rs:154-170` requires all routed expert
  tensors for `0..config.n_routed_experts` plus shared expert tensors.
- `infer/src/model/deepseek/reference.rs:515-562` CPU reference parity oracle
  executes routed experts and then adds the shared expert.

Therefore the local DSv4 1B init checkpoint has no dense MLP path. A
shared-expert-only shortcut would not be parity-equivalent to the CPU oracle,
because the oracle adds two routed experts per token plus the shared expert.
Implementing routed MoE is Phase 2B and outside this tranche.

### Gate 2: SW attention is not wired

Source evidence:

- `infer/src/model/deepseek/mla.rs:55-65` still has `todo!("DeepSeek V4
  attention kernel -- Phase 2A")` for both prefill and decode.
- `infer/src/model/deepseek/mlp.rs:37-41` still has `todo!("DeepSeek V4 MoE
  primitive -- Phase 1/2A")`.
- `infer/src/model/deepseek/forward.rs:79-93` still returns after token-range
  validation and sequence-length advance. It does not call attention, MLP,
  final norm, or the head projection.
- `infer/src/model/deepseek/weights.rs:117-149` still constructs a model shell
  with `embed_tokens=None`, `lm_head=None`, `norm=None`, `head_hc=None`, and
  `layers=Vec::new()`.

So the Phase 0 "SW-only" path is a shape/finite shell, not a true SW attention
dispatch. Wiring a real loader alone cannot make parity pass; the forward path
would immediately require new DSv4 attention and MoE execution.

## Fix

Stopped before touching runtime code. No loader, forward path, CUDA kernel, or
parity test was committed.

Do not proceed by loosening parity tolerance or by using shared-expert-only as
a hidden approximation. The next licensed implementation must be split into
smaller falsifiable units:

1. Phase 1.5.A: implement a real single-token DSv4 sliding-window attention
   boundary for the three `compress_ratio=0` layers, compared against CPU
   reference layer dumps.
2. Phase 1.5.B: implement or explicitly license routed MoE for one token
   (`gate.weight`, hash/bias routing, top-k normalization, routed experts,
   shared expert).
3. Phase 1.5.C: only after both gates are real, load all weights and re-enable
   the parity test with the original `5e-2` abs / `5e-3` rel tolerance and
   strict top-1 equality.

## Rule

Reuse-first does not license approximating a routed-MoE model as dense or
shared-expert-only when the oracle executes routed experts. Before porting a
loader, verify the model has an executable forward plan that stays inside the
tranche boundaries.
