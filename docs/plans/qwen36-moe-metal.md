# Plan: Qwen3.6-35B-A3B MoE support on Metal

**Target:** `/v1/chat/completions` serves `mlx-community/Qwen3.6-35B-A3B-4bit` on Apple Silicon via agent-infer, with bench entry.

**Hardware:** M4 Pro / 48GB unified memory. Weights ~20.4GB. Download in progress.

## Architectural finding

`mlx-lm/mlx_lm/models/qwen3_5.py` is the ground-truth reference (MIT, 531 LOC).
Qwen3.5 and Qwen3.6 share the **same Rust-visible class** in mlx-lm — MoE vs dense
is a pure config switch inside `DecoderLayer`:

```python
if args.num_experts > 0:
    self.mlp = SparseMoeBlock(args)
else:
    self.mlp = MLP(args.hidden_size, args.intermediate_size)
```

Therefore we **extend `qwen35`** rather than create a parallel `qwen36`. Config
adds 6 optional MoE fields (defaults preserve dense Qwen3.5). Attention, RoPE,
KV cache, tokenizer, scheduler, HTTP, weight prefix detection — **all reused**.

Qwen3.6 weight naming delta (from `qwen3_5_moe.py sanitize`): fused
`experts.gate_up_proj` + `experts.down_proj` → split to `switch_mlp.{gate,up,down}_proj`.

MoE block (46 lines Python, `Qwen3NextSparseMoeBlock` in `qwen3_next.py:308-354`):

```
gates   = softmax(linear(x, router))              # [B,S,E]
inds, sc = top_k(gates, k=8)                      # argpartition + take_along_axis
sc      = sc / sc.sum(-1) if norm_topk_prob
y       = SwitchGLU(x, inds)                      # MLX primitive (batched quantized)
y       = (y * sc[...,None]).sum(-2)
shared  = SwiGLU(x, shared_{gate,up,down})
shared  = sigmoid(linear(x, shared_expert_gate)) * shared
return y + shared
```

Router `gate` and `shared_expert_gate` → 8-bit quantization;
experts + shared expert → 4-bit; `group_size=64`, affine mode.

## Scope — Metal only for Phase 1

CUDA: `todo!("GPU required: Qwen3.6 CUDA not yet implemented")` following the
project pattern. M4 Pro deliverable does not need CUDA. Phase 2 follows when
validated on a Linux/NVIDIA box.

## File-level diff map

### Add (new files)

| Path | Purpose | LOC |
|---|---|---|
| `infer/src/model/qwen35/moe_config.rs` *(optional split)* | Keep MoE-specific Config fields isolated if `config.rs` grows ugly; otherwise inline into `config.rs`. | ~50 |
| `crates/mlx-sys/src/mlx_qwen35_moe_block.cpp` | C++ MoE block: `mlx_qwen35_moe_forward(hidden, weights)` wrapping `mx::argpartition`, `mx::take_along_axis`, `mx::gather_qmm`, `mx::sigmoid`, `swiglu`. Called per MoE layer. | ~300 |

### Edit

| Path | Change | LOC Δ |
|---|---|---|
| `infer/src/model/qwen35/config.rs` | Add `moe: Option<MoeConfig>` with `num_experts`, `num_experts_per_tok`, `decoder_sparse_step`, `shared_expert_intermediate_size`, `moe_intermediate_size`, `norm_topk_prob`, `mlp_only_layers: Vec<usize>`. Add `is_moe_layer(idx) -> bool` and `is_moe()` helpers. Accept `Qwen3_5MoeForConditionalGeneration` architecture string. Unit tests: load Qwen3.5-4B dense config still passes; load Qwen3.6 config populates MoE fields. | +80 |
| `infer/src/backend/metal/qwen35.rs` | (1) Replace `layer.mlp_inputs: MlpInputProjection` with `layer.mlp: MlpKind { Dense(MlpInputProjection + down_proj), Moe(MoeWeights) }`; move current dense path under `Dense`. (2) In `qwen35_forward_step` / batched paths, dispatch: `match &layer.mlp { Dense(d) => existing swiglu; Moe(m) => mlx_qwen35_moe_forward(x, m) }`. (3) `load_qwen35_metal_weights`: when config.is_moe(), load per-layer `mlp.gate.weight` (router, 8bit), `mlp.switch_mlp.{gate,up,down}_proj.weight+scales+biases` (stacked `[E, out, in/8]`, 4bit), `mlp.shared_expert.{gate,up,down}_proj.weight*`, `mlp.shared_expert_gate.weight` (8bit). Handle sanitize: split `experts.gate_up_proj` if still fused. | +400 |
| `infer/src/backend/metal/weights.rs` | Add `StackedQuantized { weight: MlxArray[E,out,in/8], scales, biases, group_size, bits }` variant on WeightTensor (or new `QuantizedExpertStack` type). Loader `load_stacked_quantized(prefix) -> StackedQuantized`. | +120 |
| `crates/mlx-sys/src/mlx_bridge.cpp` + `.h` | Expose `mx::argpartition`, `mx::take_along_axis` (bind if not present), `mx::gather_qmm` (batched quantized matmul taking int indices + quantized stack), `mx_switch_glu` composite helper. | +180 |
| `crates/mlx-sys/src/lib.rs` | `extern "C"` FFI declarations for the new bridge functions. | +60 |
| `crates/mlx-sys/build.rs` | Add `mlx_qwen35_moe_block.cpp` to cc::Build sources. | +2 |
| `infer/src/model_registry.rs` | Add `Qwen3_5_Moe` to `ModelArch` enum; map architectures `Qwen3_5MoeForConditionalGeneration` and `Qwen3_5MoeForCausalLM` to it; `attention_variant` returns hybrid; `is_implemented` = metal-only. | +15 |
| `infer/src/backend/cuda/bootstrap.rs` | Add `Qwen35Moe` to `ModelType` enum; `detect_model_type` routes Qwen3_5_Moe → Qwen35Moe; CUDA `load_qwen35_moe_components` = `todo!("GPU required: Qwen3.6 CUDA not yet implemented")`. | +15 |
| `infer/src/server_engine.rs` | Add `Qwen35Moe(Qwen35InferenceEngine)` variant (reuse Qwen35InferenceEngine since arch is superset); dispatch arm. | +20 |
| `crates/cli/src/model_catalog.rs` | Register `mlx-community/Qwen3.6-35B-A3B-4bit` discovery hint if pattern-driven. | +5 |

**Total estimated delta: ~1200 LOC.**

### Stub (CUDA)

`infer/src/model/qwen35/{forward,prefill,decode,batch_decode,weights}.rs`:
no edits. MoE dispatch lives on Metal side only. When running `cargo build`
with `--features cuda` and Qwen3.6 is requested, bootstrap rejects with the
`todo!` message.

## Weight key mapping (MLX Qwen3.6 → agent-infer)

Reference mlx-lm `qwen3_5_moe.py:sanitize()`:

```
model.layers.N.mlp.experts.gate_up_proj   [E, 2*Hmoe, H/8]  → split to:
  model.layers.N.mlp.switch_mlp.gate_proj.weight [E, Hmoe, H/8]
  model.layers.N.mlp.switch_mlp.up_proj.weight   [E, Hmoe, H/8]
model.layers.N.mlp.experts.down_proj      [E, H, Hmoe/8]    → switch_mlp.down_proj.weight

model.layers.N.mlp.gate.weight            [E, H/8]          — router, 8bit
model.layers.N.mlp.shared_expert.{gate,up}_proj.{weight,scales,biases}
model.layers.N.mlp.shared_expert.down_proj.{weight,scales,biases}
model.layers.N.mlp.shared_expert_gate.weight                [1, H/8] — 8bit
```

agent-infer prefix detection (`backend/metal/qwen35.rs:1563`) already tries
`language_model.model`, `model.language_model`, `model` — same logic applies.

## Acceptance

- [ ] `cargo test --release --no-default-features --features metal` green
- [ ] `cargo check -p infer --no-default-features --features cuda,no-cuda` typechecks (CUDA `todo!` stub does not block this)
- [ ] `cargo clippy --no-default-features --features metal -- -D warnings` clean
- [ ] Smoke: `mlx-community/Qwen3.6-35B-A3B-4bit` generates coherent tokens for prompt "解释一下什么是 Mixture of Experts:"; first 64 tokens match `mlx_lm.generate --temp 0` byte-for-byte (or document drift source)
- [ ] `scripts/bench_guidellm.sh qwen36-4bit-metal` completes; `docs/experience/wins/2026-04-20-qwen36-moe-metal-landing.md` lands with Δ vs Qwen3.5-4B-MLX-4bit
- [ ] `docs/support-matrix.md` §3 adds Qwen3.6 row (Beta, Metal-only)

## Delegation plan

1. **General-purpose subagent A — Config + registry** (~130 LOC, 3 files, no deps). Fire first.
2. **General-purpose subagent B — MLX bridge additions** (~240 LOC, 3 files, no deps on A). Fire parallel to A.
3. **General-purpose subagent C — Metal weights + forward** (~520 LOC, 2 files). Depends on A & B — fire after both report back.
4. **Claude integrates** — routing/registry wiring, CUDA stubs, smoke test, bench.
5. **`codex review --uncommitted`** before commit.

## Open questions (flag if surfaced during implementation)

1. Does MLX C++ `gather_qmm` primitive exist at our pinned MLX version? If not, falls back to per-expert quantized gemv loop (slower but correct) — benchmark before optimizing.
2. KV cache sizing at 48GB with 20GB weights: if context > 32K tokens causes OOM, cap default `max_seq_len` at 32K for Qwen3.6 and document.
3. The `mlp_only_layers` field is empty in Qwen3.6-35B-A3B config (all 40 layers are MoE) — verify when config lands locally.
