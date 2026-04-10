# 2026-04-10 · Qwen3.5 GGUF partial-RoPE cos cache stride bug

> **Update — the real bug.** The initial root-cause below (`attn_output_gate`
> missing) turned out to be wrong: the gate was already wired correctly in
> both the prefill and batch-decode paths via the fused
> `prefill_attention_hd256_prep_cuda` +  `attention_gate_batch_hd256_cuda`
> kernels. The actual bug was a one-line discrepancy in `precompute_rope`
> arguments between the safetensors and GGUF loaders:
>
> ```rust
> // safetensors (correct)
> precompute_rope(&ctx, config.rotary_dim, 4096, config.rope_theta)?;
> // GGUF (broken — was head_dim, stride=256)
> precompute_rope(ctx, config.head_dim, 4096, config.rope_theta)?;
> ```
>
> The CUDA kernel indexes `cos_cache[pos * rotary_dim + d]` (stride=64),
> but the GGUF path built the cache with stride=256. For every position > 0,
> the kernel read values from position 0's latter half — which is
> `cos(0 * inv_freq[i]) = 1.0` for all i — so every rotated Q/K element
> got cos=1, sin=0. That's effectively **zero position encoding across the
> entire sequence**, which collapses attention to a position-independent
> average and produces the prompt-independent degenerate output we observed.
>
> Fix: change one argument to `config.rotary_dim` in `qwen35/weights.rs:692`.
>
> **Validation**: Qwen3.5-4B Q6_K now generates coherent text:
> `"The capital of France is"` → `" Paris.\nA. True\nB. False..."`.
> Previously it produced the same repeating-token garbage as Carnice-27b.
>
> **Remaining bugs** found during this investigation (separate fixes):
> - Qwen3.5-4B Q4_K_M still fails (BF16 fallback also fails → bug in
>   `dequantize_row_q4_K` in `gguf.rs`, not in the native GEMV kernel).
> - Carnice-27b Q6_K still fails — Carnice has `num_v_per_k = 48/16 = 3`
>   while Qwen3.5-4B has `vpk = 2`, and our `reverse_v_reorder_rows` may be
>   wrong for `vpk = 3`.

---

## Original (incorrect) root cause below — kept for audit trail

# 2026-04-10 · Qwen3.5 full-attention output gate dropped on the floor

## Context

Carnice-27b (Qwen3.5 arch, 64 layers, 16 full-attention + 48 linear-attention)
loaded from Q4_K_M GGUF produced completely degenerate output: every prompt,
every length, every sampling config → one repeating token (`"零食零食..."` /
`"不懂不懂..."`). Same symptom on Qwen3.5-4B. Bisect showed:

- Dequant kernels match llama.cpp Python reference byte-for-byte.
- `PEGAINFER_FORCE_BF16_QUANT=1` (skip all packed GPU kernels, dequant at load
  to BF16) → identical garbage. Rules out Q4_K GEMV kernel bugs.
- `PEGAINFER_QWEN3_FP32_RESIDUAL=1` (fp32 residual shadow across all 36 layers
  of Qwen3-4B) → identical garbage. Rules out bf16 residual accumulation.
- Output is completely prompt-independent — even a 2300-token prompt vs an
  8-token prompt produce the same tokens. Points to degenerate final hidden
  state, not a length/position/chunked-prefill bug.

## Root Cause

Qwen3.5 (Qwen3-Next) has `attn_output_gate: true` in its config. The
`q_proj` output dim is `num_heads * head_dim * 2`, packed as a **per-head
concat** `[q_head_0 | gate_head_0 | q_head_1 | gate_head_1 | ... ]`. The
correct formula is:

```python
# HF transformers (modeling_qwen3_next.py)
q_gate = self.q_proj(x).view(*input_shape, num_heads, head_dim * 2)
q, gate = torch.chunk(q_gate, 2, dim=-1)
# ...attention...
attn_output = attn_output * torch.sigmoid(gate)
```

Confirmed identical in llama.cpp (`src/models/qwen3next.cpp`, `ggml_view_4d`
with stride `nb[1] = 2*head_dim*elem_size`, gate offset `head_dim*elem_size`)
and vLLM (`qwen3_next.py` L253-318). All three agree: **per-head concat layout,
sigmoid activation**.

Our `qwen35/prefill.rs:147` does:

```rust
let q_full_batch = ops::gemm(&self.ctx, &attn.q_proj, normed_batch)?;
// q_full_batch shape = [2*n_heads*head_dim, seq_len] (per-head [q|g|q|g|...])

ops::prefill_attention_hd256_batch(..., &q_full_batch, ...,
    num_attention_heads=24, ...)
// attention kernel reads the first num_heads*head_dim=6144 elements as 24 Q heads
```

Under the correct **per-head** layout, the first 6144 elements are
`[q_0, gate_0, q_1, gate_1, ..., q_11, gate_11]` — **12 real Q heads
interleaved with 12 gate heads, fed to the attention kernel as if they were
24 Q heads**. The attention output is also never multiplied by
`sigmoid(gate)`. Two structural bugs, both silent.

This explains every observed symptom:

| Symptom | Explained by |
|---|---|
| Prompt-independent garbage | Q is half noise → attention collapses identically regardless of input |
| All layers degraded | Every one of the 16 full-attn layers has the same bug |
| fp32 residual shadow no help | Bug is upstream of residual |
| V row reorder irrelevant | Bug is in Q, not V |
| Q4_K and BF16 fallback both fail | Bug is in forward pass, not load path |
| Qwen3-4B still fails (but differently) | Qwen3-4B has a *different* bug — no output gate in its config. Separate investigation. |

## Fix

Two-part structural fix touching prefill + decode + Metal backend:

1. New CUDA kernel `split_q_gate_batch` that de-interleaves
   `q_full [2*n_h*hd, seq]` into `q_only [n_h*hd, seq]` and
   `gate [n_h*hd, seq]`. Simple strided copy, one thread per output element.
2. New fused `sigmoid_mul_batch` kernel: `out[i] = x[i] * sigmoid(g[i])`.
3. Wire into `qwen35/prefill.rs` full-attention path: GEMM → split →
   attention (on `q_only`) → `sigmoid_mul(attn_out, gate)` → o_proj.
4. Same wiring for batch decode + single-token decode.
5. Mirror in Metal backend.

Gate buffer allocation added to prefill/decode/single-token buffer structs.
No weight reordering at load time — keep q_proj packed in Q4_K.

## Rule

**When porting a model, compare the config schema field-by-field against
HF, not just the hidden dims.** `attn_output_gate`, `partial_rotary_factor`,
`mrope_interleaved` / `mrope_section` were all present in the Carnice-27b
config.json and all silently ignored by our loader. A skipped bool flag in
a config struct can bake in a multi-hour structural bug that unit tests on
kernels will never catch.

**If a model outputs prompt-independent garbage, stop hunting in kernels and
compare against HF's forward() line-by-line for the specific architecture.**
Kernel bugs usually produce noisy-but-responsive output; structural bugs
(dropped projection halves, skipped activations, wrong layer wiring) are
what produces degenerate constant-token outputs.
