# Plan — FP8 KV per-channel K + per-token V (KIVI scheme) for Qwen3-dense

**Status**: Draft · **Driver**: ckl · **Trigger**: 2026-05-26 audit confirmed FP8
catastrophic on Qwen3-4B is the precision-floor compounding signature KIVI
solved.

## Why

The 2026-05-26 cross-precision parity audit
(`infer/tests/kv_precision_parity.rs`) plus the multi-layer attn-out dump on
A100 sm_80 confirmed FP8 E4M3 KV produces `mean_match = 0.0156` on Qwen3-4B
dense (36 full-attention layers) while INT8 with the same per-(token, head)
8-bit quant gets `mean_match = 1.0`. Root cause: K has channel-wise outliers,
per-(token, head) absmax gets dominated by those outliers, the other channels
underflow toward zero, BF16 attn-output write truncates the tiny softmax×V
products to zero, divergence compounds through depth (K-scales shrink 10×
layer 0 → layer 17).

The literature solved this exact problem:

- **KIVI** ([arXiv 2402.02750](https://arxiv.org/abs/2402.02750), ICML 2024):
  per-channel scale for K + per-token scale for V. On Llama-2-13B at 2-bit
  KV, lifts CoQA from 2.88 (per-token K, collapsed) to 63.53 (per-channel K).
  Reduces attention-score error 5× vs per-token K.
- **KVLinC** ([arXiv 2510.05373](https://arxiv.org/abs/2510.05373), Oct 2025):
  Hadamard-rotate K before per-channel quant. Evaluated on Qwen3-4B
  specifically — +6.4% GSM8K, +10% RULER over KIVI baseline.

vLLM's per-tensor FP8 KV ([blog](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html))
ships unresolved divergence on the same family of models (
[SGLang #22671](https://github.com/sgl-project/sglang/issues/22671),
[vLLM RFC #37319](https://github.com/vllm-project/vllm/issues/37319)). The
per-head scheme added in [vLLM PR #30141](https://github.com/vllm-project/vllm/pull/30141)
is closer to KIVI but still not per-channel. ARLE can leapfrog by going
straight to KIVI's storage scheme.

## What to change

### Step 1 — Storage layout: K scale per (kv_head, head_dim), V scale per (token, kv_head)

Currently:

```
K scales: [total_rows, num_kv_heads]      ← per (token, head)
V scales: [total_rows, num_kv_heads]      ← per (token, head)
```

KIVI:

```
K scales: [num_kv_heads, head_dim]        ← per (head, dim) — PER-LAYER, NOT per-row
V scales: [total_rows, num_kv_heads]      ← unchanged (per token, head)
```

K-scale capacity drops from `O(max_total_tokens × num_kv_heads)` to
`O(num_kv_heads × head_dim)` per layer. For Qwen3-4B (8 KV heads, head_dim=128,
~50K tokens): 50000×8 = 400K floats → 8×128 = 1024 floats per layer. **400×
fewer scales for K**.

But K-scale becomes **static per layer** (computed once, used for all K writes
in that layer). Need calibration to compute it. Two options:

- **Dynamic running absmax** (KIVI's approach): track running max per
  `(kv_head, head_dim)` channel across all tokens written so far. Update
  online; rescale FP8 bytes on each scale bump. Risky for streaming
  (re-quantization storms).
- **Calibration-based static scale**: run a short corpus once at server
  boot, compute per-channel absmax over the calibration set, freeze. This is
  what vLLM PR #30141 does for per-head; we'd extend to per-channel.

Recommend **static calibration** for V1: simpler, no online re-scaling, matches
vLLM/SGLang ergonomics. Calibration script loads model + 32 short prompts,
records per-(kv_head, head_dim) absmax for K, per-(token, kv_head) absmax
distribution for V, writes a `.safetensors` file alongside the model weights.
Server reads it at boot.

### Step 2 — Kernel changes

`quantize_paged_kv_fp8_kernel` (`crates/cuda-kernels/csrc/kv/kv_quant.cu:171`):
no per-(token, head) reduction needed for K. Just `kv_fp8[dst] = (val *
inv_scale[kv_head, d])` using the pre-computed scale. Simpler kernel.

`quantize_scatter_kv_fp8_kernel` (migration path): same change.

`decode_attention_fp8_partial_kernel`
(`crates/cuda-kernels/csrc/attention/decode_attention_quantized.cu:307`):
K_scales pointer changes shape from `[total_rows, num_kv_heads]` to
`[num_kv_heads, head_dim]`. Per-warp loop reads one scalar `k_scale[h, d]`
instead of one per token. **Memory traffic for scales drops** (no per-token
scale load). V-side unchanged.

### Step 3 — Calibration script

`scripts/calibrate_fp8_kv.py`:

1. Load model BF16
2. Forward 32 prompts × 256 tokens through layers
3. For each `(layer, kv_head, head_dim_channel)`, record absmax of K values
4. For each `(layer, kv_head)`, record absmax distribution of V values (use
   p99.9 to clip outliers)
5. Write `infer/models/Qwen3-4B/fp8_kv_calibration.safetensors`:
   - `k_scales`: `[num_layers, num_kv_heads, head_dim]` f32
   - `v_scales_p999`: `[num_layers, num_kv_heads]` f32 (currently unused but
     could enable V per-channel later)

Server boot: if calibration file present, load + use. Else fail with
`--kv-cache-dtype fp8 requires <model>/fp8_kv_calibration.safetensors;
generate with scripts/calibrate_fp8_kv.py <model>`.

### Step 4 — Verify with parity audit

Re-run `kv_precision_parity.rs` audit on Qwen3-4B with the new FP8 path.
Target `mean_match >= 0.95` at 64-token horizon. If achieved, flip the
harness's FP8 gate from `None` to `Some(0.95)`.

## Risk and tradeoffs

- **Calibration drift**: scales computed on calibration corpus may not
  generalize. Mitigation: use a diverse calibration set (code + chat + math
  prompts). Document this caveat in calibration script.
- **No streaming-fit calibration**: model can't be served before calibration
  runs. Adds an offline step. Mitigation: ship pre-calibrated scale files
  alongside known-good Qwen3 / Qwen3.5 models.
- **K-scale lookup costs registers**: the kernel inner loop now indexes a
  `(num_kv_heads, head_dim)` table per token instead of one scalar. Modest
  shared-memory layout change; net memory traffic still drops because no
  per-token scale load.

## Acceptance

- Audit parity for FP8 on Qwen3-4B: `mean_match >= 0.95` at 64-token horizon.
- Audit parity for FP8 on Qwen3.5-4B: `mean_match >= 0.95` (currently 0.66
  prompt-drift).
- Kernel diagnostics in `crates/cuda-kernels/src/kv_quant.rs` (the three
  existing roundtrip + kernel-pair tests) still pass after the storage-
  layout change.
- Memory savings vs BF16 KV preserved (1B/value FP8 + small scale table).
- No regression on INT8 / BF16 paths.

## Cross-refs

- Trigger: [`docs/experience/errors/2026-05-26-fp8-kv-step1-divergence-known-deferred.md`](../experience/errors/2026-05-26-fp8-kv-step1-divergence-known-deferred.md)
- Audit framework: [`docs/plans/2026-05-25-kv-precision-parity-framework.md`](2026-05-25-kv-precision-parity-framework.md)
- KIVI: https://arxiv.org/abs/2402.02750
- KVLinC: https://arxiv.org/abs/2510.05373
- vLLM per-head PR: https://github.com/vllm-project/vllm/pull/30141
- SGLang catastrophic FP8 KV issue: https://github.com/sgl-project/sglang/issues/22671
