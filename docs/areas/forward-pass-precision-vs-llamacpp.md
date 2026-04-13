# Forward-Pass Precision Comparison: agent-infer vs llama.cpp (ggml)

## Context

Our engine runs Qwen3-4B safetensors fine but produces garbage on the
same Qwen3-4B loaded from `.gguf` (Q4_K_M). llama.cpp runs the same
GGUF file and produces correct text. All dequant paths (Q3_K/Q4_K/Q5_K/
Q6_K/Q8_0) are proven bit-exact against a Python llama.cpp reference.
The bug is in the forward pass, not in weight loading.

This doc is a kernel-by-kernel precision comparison between our Qwen3
forward pass and llama.cpp's `llm_build_qwen3` (from
`src/models/qwen3.cpp`), to bound what can drift between the two.

## llama.cpp Qwen3 layer body (verbatim, from llm_build_qwen3)

```cpp
// norm
cur = build_norm(inpL, attn_norm, NULL, LLM_NORM_RMS, il);

// self-attention
{
    Qcur = build_lora_mm(wq, cur);
    Kcur = build_lora_mm(wk, cur);
    Vcur = build_lora_mm(wv, cur);

    Qcur = reshape_3d(Qcur, n_embd_head, n_head,    n_tokens);
    Kcur = reshape_3d(Kcur, n_embd_head, n_head_kv, n_tokens);
    Vcur = reshape_3d(Vcur, n_embd_head, n_head_kv, n_tokens);

    Qcur = build_norm(Qcur, attn_q_norm, NULL, LLM_NORM_RMS, il);
    Qcur = ggml_rope_ext(Qcur, inp_pos, ..., n_rot, rope_type, ...);

    Kcur = build_norm(Kcur, attn_k_norm, NULL, LLM_NORM_RMS, il);
    Kcur = ggml_rope_ext(Kcur, inp_pos, ..., n_rot, rope_type, ...);

    cur = build_attn(wo, bo, Qcur, Kcur, Vcur, ..., 1.0f/sqrtf(n_embd_head), il);
}

ffn_inp = ggml_add(cur, inpSA);                  // post-attention residual
cur = build_norm(ffn_inp, ffn_norm, NULL, LLM_NORM_RMS, il);
cur = build_ffn(cur, ffn_up, ffn_gate, ffn_down, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
cur = ggml_add(cur, ffn_inp);                    // MLP residual
```

## Step-by-step comparison

| # | llama.cpp | agent-infer (qwen3/prefill.rs forward_layer_batch) | Semantic match? | Precision match? |
|---|-----------|------|-----|-----|
| 1 | `build_norm(inpL, attn_norm, RMS)` | `rms_norm_batch_into(hidden, input_layernorm)` | ✓ | **was NO: rounded to bf16 between `x*inv_rms` and `*weight`** — now fixed to stay in fp32 until the final store |
| 2 | `wq @ cur` | `gemm_into(q_proj, normed → q_batch)` | ✓ | ✓ cuBLAS bf16 × bf16 with fp32 accumulator |
| 3 | `wk @ cur` | `gemm_into(k_proj, normed → k_batch)` | ✓ | ✓ |
| 4 | `wv @ cur` | `gemm_into(v_proj, normed → v_batch)` | ✓ | ✓ |
| 5 | reshape Q/K/V to `[head_dim, n_head, n_tokens]` | implicit via head-strided indexing in the prep kernel | ✓ | n/a |
| 6 | `build_norm(Qcur, attn_q_norm, RMS)` | inside `prefill_qk_norm_rope_kernel`: per-head RMS + `*q_norm_weight` | ✓ | **was NO: rounded to bf16 between scale and weight, then rounded AGAIN into bf16 smem for RoPE** — now fixed to full fp32 smem and no intermediate rounding |
| 7 | `ggml_rope_ext(Qcur, ...)` | same kernel continues: RoPE pair rotation via smem | ✓ (NEOX half-split) | now matches (fp32 smem) |
| 8 | `build_norm(Kcur, attn_k_norm, RMS)` | same kernel, K branch | ✓ | same as step 6 |
| 9 | `ggml_rope_ext(Kcur, ...)` | same kernel continues | ✓ | same as step 7 |
| 10 | `build_attn(wo, bo, Q, K, V, 1/√d_head)` | `flashinfer_single_prefill(q, k_cache, v_cache, ..., sm_scale=1/√d_head)` + separate `gemm_into(o_proj, attn_output)` | ✓ mathematically; ours splits attention from o_proj (llama.cpp fuses) | **presumed ✓**: flashinfer uses fp32 softmax accumulators. NOT verified at source. bo (o_proj bias) = NULL for Qwen3, so the fact we don't support it is OK. |
| 11 | `ggml_add(cur, inpSA)` | `add_batch_into(hidden, o_buf → hidden_out)` + swap | ✓ | bf16 add (lossy). llama.cpp activations are f32 by default → f32 add. **Difference.** |
| 12 | `build_norm(ffn_inp, ffn_norm, RMS)` | `rms_norm_batch_into(hidden, post_attention_layernorm → normed)` | ✓ | same as step 1 |
| 13 | `build_ffn(cur, ffn_up, ffn_gate, ffn_down, SILU, PAR)` | `gemm_into(gate_proj)`; `gemm_into(up_proj)`; `silu_mul_batch_into(gate, up → act)`; `gemm_into(down_proj)` | ✓ | GEMMs: ✓. silu_mul: **was NO — triton kernel rounded `silu` to bf16 before `* up`**, now fixed to `(silu * up).to(bf16)` |
| 14 | `ggml_add(cur, ffn_inp)` | `add_batch_into(hidden, o_buf → hidden_out)` | ✓ | bf16 add (same as step 11) |

## Root cause structure

Three precision losses found and fixed in this pass:

1. **`rms_norm_batched_kernel`** (`norm.cu`): was rounding `x * inv_rms`
   to bf16 before multiplying by weight. Fixed to compute
   `x * inv_rms * weight` fully in fp32 with one bf16 store at the end.

2. **`prefill_qk_norm_rope_kernel`** (`prefill_attention.cu`): was
   rounding the per-head normed value to bf16 before weight multiply,
   and again rounding into bf16 shared memory before RoPE pair rotation.
   Fixed to fp32 throughout (including a fp32 smem of 512 bytes).

3. **`silu_mul_kernel`** (`tools/triton/silu_mul_kernel.py`): was
   casting `silu(gate)` to bf16 before multiplying by `up`. Fixed to
   `(silu * up).to(bf16)`.

Each of these is a ~0.4%-per-element precision loss. They compound
per layer, and when combined with Q4_K's ~20% per-tensor RMS noise,
push the forward pass into a pathological regime that manifests as a
catastrophic explosion at layer 5 (rms 0.42 → 16 in a single layer).

## Remaining precision differences (not yet addressed)

The fixes above were necessary but NOT sufficient — the Qwen3-4B GGUF
smoke test still produces stuck tokens. The remaining structural gaps
between our path and llama.cpp:

- **Activation dtype across layers.** llama.cpp ggml uses fp32 tensors
  for activations; we use bf16 `HiddenStates`. Every residual add,
  every GEMM input, every attention Q/K/V is bf16 on our side, fp32
  on theirs. This is a systemic ~0.4% precision floor on every
  inter-kernel boundary.

- **`add_batch_into`** (residual adds) operates in bf16 — cannot
  preserve fp32 precision that a hypothetical upstream fp32 kernel
  would produce. Even if a norm kernel writes fp32 internally, the
  bf16 storage for the next kernel's input truncates.

- **flashinfer attention precision.** Unverified at source whether
  the inner softmax accumulator is fp32 (it should be), and whether
  the mask / scale are applied before or after the softmax subtract-
  max step.

- **o_proj split from attention.** llama.cpp's `build_attn` fuses
  attention compute + o_proj; we do them as separate kernels with a
  bf16 handoff. Functionally equivalent but adds one more bf16
  round-trip.

## Follow-ups to investigate

1. **Promote `HiddenStates` to fp32 end-to-end** (expensive — touches
   every kernel signature). Verify against safetensors e2e that this
   doesn't change correct outputs, then check GGUF L5 explosion.

2. **Dump flashinfer intermediate** (softmax scores, V @ softmax
   output) for layer 5, compare to a Python reference that runs the
   same Q/K/V values through pytorch F.scaled_dot_product_attention.
   Pinpoints whether the issue is inside flashinfer or in the
   activations it's fed.

3. **Per-element error accumulation**. Instrument an element-wise
   diff at each checkpoint (not just rms/head[0..8]) to see which
   dimensions drift first.

4. **llama.cpp hidden state hook**. Use `llama-cpp-python` with
   `logits_all=True` and a forward hook equivalent to dump its layer-5
   activations for the same prompt, bit-compare against ours.
