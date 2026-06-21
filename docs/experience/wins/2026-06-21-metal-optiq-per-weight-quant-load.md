# OptiQ (mixed-precision) loads on arle Metal — per-weight quant config + mixed gate/up merge

## Context
`mlx-community/Qwen3.6-27B-OptiQ-4bit` (OptiQ mixed 4/8-bit) failed to build the serve router on
the Metal backend. Two stacked blockers, both because the loader assumed one global quant config.

## What Worked
1. **Per-weight quant config.** OptiQ's `config.json` `quantization` dict carries a global
   `{group_size:64, bits:4}` PLUS per-weight overrides (object-valued entries keyed by the full
   tensor name, e.g. `"...embed_tokens": {bits:8, group_size:64}`). `QuantConfig` gained
   `per_weight: Arc<HashMap<String,(i32,i32)>>` (parsed in `config.rs`); `loader.rs`
   `load_proj_from_tensors`/`load_embed_tokens_from_tensors` look up `base` there, falling back to
   the global. Fixes the `mlx_dequantize` scales/biases shape mismatch.
2. **Mixed-bit gate/up merge.** 18 dense-MLP layers have `gate=4-bit, up=8-bit`; the merged
   `gate_up` path can't row-concat mixed formats. `weights.rs::concat_weight_rows` now dequantizes
   both to dense bf16 and concatenates → the C++ merged `mlp()` consumes it via its dense fallback.
   Pure Rust; no C++ change.

Verified (build clean, 30 infer-metal tests pass): OptiQ now loads (~10s) and decodes correctly
("The capital of France is **Paris**", primes). **Plain 4bit did NOT regress: 14.71 tok/s** (the
compiled-MLP win survives), full 63104-token KV.

## Caveat (measured, stated — not a silent win)
OptiQ runs **7.96 tok/s vs plain 4bit's 14.71 (−46%)**: the 18 dequantized gate/up layers are
dense bf16 → heavier memory reads AND ineligible for `compiled_mlp_fn` (it gates on quantized).
Weights 18 GB (vs 14), KV 17808 tokens (vs 63104). Quality is *higher* on those layers (bf16 > the
8-bit it replaced), but the speed cost is real. **Plain 4bit stays the speed default; OptiQ is a
quality option.** Speed follow-up = a C++ separate-MLP forward path so the 8-bit `up_proj` stays
quantized instead of dequantizing to bf16.

## Rule
- **Verify a delegated win, don't trust the summary.** The subagent reported KV=480 (a transient
  worst case); the real measured budget is 17808 — and the −46% speed cost only showed up under a
  measured A/B, not in the "it loads + is coherent" summary. A "loads correctly" win with a
  half-speed caveat is a *partial* win — state the number.
