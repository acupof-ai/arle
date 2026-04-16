# 2026-04-10 · Remaining GGUF bugs after RoPE stride fix

Tracking the three independent GGUF-path bugs still outstanding after the
RoPE stride fix (`precompute_rope` in `qwen35/weights.rs`, commit `67a838f`).
Each has a separate root cause — knocking one down will not help the others.

## Bug 1 · `dequantize_row_q4_K` in `gguf.rs` is wrong

**Symptom**: Qwen3.5-4B Q4_K_M (and Q4_K_S) produce degenerate output
(`"back back back..."`) even with `INFER_FORCE_BF16_QUANT=1`, which
skips the native Q4_K GEMV kernel entirely and uses CPU-side dequant +
BF16 upload.

**Evidence the kernel is not at fault**:
- `INFER_FORCE_BF16_QUANT=1` forces `gguf.read_tensor_bf16()` → BF16
  `DeviceMatrix::from_host()`, skipping `from_quantized_q4k`.
- Same garbage output with and without the env var.
- Qwen3.5-4B Q6_K (`dequantize_row_q6_K`) works fine through the same
  fallback — so the issue is Q4_K-specific.

**Where to look**: `infer/src/gguf.rs::dequantize_row_q4_K`. The function
passed the `ground_truth_q4k.rs` test against a Python reference, so the
bug is likely:
- A boundary / block-ordering corner case not covered by the synthetic
  test (e.g. the last super-block of a row that's not a multiple of 256).
- A subtle `dmin` bias error that cancels on mean-zero synthetic inputs
  but matters on real weights.

**First action**: Port llama.cpp's `dequantize_row_q4_K` from
`ggml/src/ggml-quants.c` byte-for-byte and diff our function against it
on the same real Carnice-27b Q4_K tensor, row by row, not just block 0.

## Bug 2 · `reverse_v_reorder_rows` may be wrong for `num_v_per_k >= 3`

**Symptom**: Carnice-27b Q6_K still produces degenerate output
(`" define define...`") even with the RoPE fix, while Qwen3.5-4B Q6_K
generates coherent English with the same fix.

**Structural difference between the two models**:

| field | Qwen3.5-4B | Carnice-27b |
|---|---|---|
| `num_hidden_layers` | 32 | 64 |
| `hidden_size` | 2560 | 5120 |
| `num_attention_heads` | 16 | 24 |
| `linear_num_value_heads` | 32 | 48 |
| `linear_num_key_heads` | 16 | 16 |
| `vpk = n_v / n_k` | **2** | **3** |

**Why this is suspect**: `reverse_v_reorder_rows`
(`weight_loader.rs:635`) permutes GGUF's `[num_v_per_k, num_k_heads,
head_dim, cols]` layout into HF's `[num_k_heads, num_v_per_k, head_dim,
cols]`. Branch:

```rust
let gguf_head = v * num_k_heads + k;
let hf_head   = k * num_v_per_k + v;
data[dst_start..dst_start + size].copy_from_slice(&src[src_start..src_start + size]);
```

For `vpk=2` this happens to produce a simple pairwise swap that could
"happen to work" even if the logical indexing were slightly off. For
`vpk=3` every non-trivial permutation cycle is length > 2 and an
off-by-one in the mapping would scramble heads. Needs byte-level
verification against llama.cpp's
`_LinearAttentionVReorderBase.modify_tensors` on an actual Carnice-27b
tensor (not synthetic).

**Also suspect**: `reverse_v_reorder_cols`, `reverse_v_reorder_f32`,
`load_tensor_1d_gguf_v_reorder`, `load_tensor_2d_gguf_v_reorder_rows`,
`load_tensor_2d_gguf_v_reorder_cols` — they all share the same
`gguf_head = v*k + k` → `hf_head = k*v + v` logic.

**First action**: Dump one Carnice-27b `in_proj_z` tensor (vpk=3) from
both (a) our loader output and (b) a Python reference that does the
exact llama.cpp permutation, then diff. If they differ, the rule is
wrong; if identical, the bug is elsewhere (e.g. the GDR kernel assumes
HF layout but gets our permuted-but-still-wrong layout).

**Alternative path**: Check whether llama.cpp's HF→GGUF converter
really reorders V at all, or whether this entire reorder step is
fictional. The earlier research (see
`2026-04-10-qwen35-attn-output-gate-missing.md` audit trail) found that
llama.cpp does NOT reorder V for full attention — only the linear-
attention variant has a dedicated reorder class, and its name
(`_LinearAttentionVReorderBase`) strongly suggests it's the GGUF-side
forward reorder, which we should then reverse. But there's a chance
our "reverse" actually matches the forward direction and we're
double-applying it. Grep llama.cpp `convert_hf_to_gguf.py` for
`_reorder_v_heads` and follow the call graph.

## Bug 3 · Qwen3-4B (non-3.5) Q4_K_M degenerate output

**Symptom**: Qwen3-4B (plain, no linear attention, 36 layers) loaded
from `Qwen3-4B-Q4_K_M.gguf` produces identical repeating garbage
(`"零食零食..."` / `"不懂不懂..."`) for any prompt. `INFER_FORCE_BF16_QUANT=1`
reproduces the same output → not the native Q4_K kernel.

**Why this is a separate bug from #1 and #2**:
- Qwen3 has no `attn_output_gate` (config doesn't have the field).
- Qwen3 has no linear attention (so no `reverse_v_reorder_rows`).
- Qwen3 uses `qwen3/weights.rs`, not `qwen35/weights.rs` — different
  `precompute_rope` call site, never had the stride bug.
- So none of the Qwen3.5 fixes can apply here.

**Suspects**, in decreasing likelihood:
1. **Also the `dequantize_row_q4_K` bug from #1.** Qwen3-4B Q4_K_M uses
   the same dequant function. If #1 is fixed, test #3 first before
   investigating further.
2. **GGUF metadata / config loading** for Qwen3: one of `rope_theta`,
   `rms_norm_eps`, `head_count_kv`, or `head_dim` is being read from
   the GGUF metadata header wrong and silently falling back to a
   Qwen2-era default. `qwen3/config.rs::from_gguf_metadata` is
   suspect-worthy.
3. **`embed_tokens` row stride** when force-loaded as BF16
   (`load_tensor_2d_gguf_bf16`) — that code path was added for the
   quantized embed_tokens case and may have a wrong row/col
   interpretation for Qwen3 vs Qwen3.5 vocab shapes.

**First action**: Test Qwen3-4B Q6_K and Q8_0 from GGUF. If they work,
the bug is squarely in the Q4_K dequant path (→ fix #1 first and
retest). If they also fail, the bug is in Qwen3's GGUF loader itself,
independent of quant format.

---

## Triage order

1. **Bug 1 first** — it's concretely localized (one function, one file)
   and likely unblocks Bug 3 for free.
2. **Then Bug 3** — retest after Bug 1 is fixed; if still broken, hunt
   the Qwen3 config-loading or embed-lookup path.
3. **Bug 2 last** — needs byte-level cross-reference against llama.cpp
   to distinguish "reorder rule wrong" from "reorder shouldn't exist".

## Rule

**When a single class of symptom (prompt-independent degenerate output)
hits multiple unrelated model paths, stop assuming one bug.** It's
cheaper to enumerate the independent code paths and bisect each one
than to keep searching for "the" root cause. Today's hunt burned hours
chasing the gate hypothesis before bisect showed the bug hit both
paths that don't share the gate logic.
