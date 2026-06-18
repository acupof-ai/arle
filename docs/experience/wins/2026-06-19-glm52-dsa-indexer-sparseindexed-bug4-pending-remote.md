# GLM-5.2 BUG4: DSA Lightning Indexer Wired for SparseIndexed (pending-remote)

## Context

GLM-5.2 (`glm_moe_dsa`) maps onto the shared DSv4 V32 runtime with
`DeepSeekV4AttentionMode::SparseIndexed` on every layer. The DSA lightning
indexer — the sparse-attention KEY SELECTION (build index q/k, weighted-ReLU MQA
logits, top-`index_topk` select → `selected`) — and the `dsa_official` selector
state were gated to `mode == CompressedSparse` only. GLM is `SparseIndexed`, so
the indexer prepass never ran, `selected` was never produced, and the
SparseIndexed FlashMLA decode/prefill (which already consume `selected`,
`mode_int=1`, `max_compressed_keys=index_topk`) had nothing to read → GLM decode
errored/garbaged. This is the core sparse-attention mechanism — the most
important Tranche-D fix.

Chain: A=03dee9b6, B=934f15b1, C=b41fa075, D=7157f721, fixes=bfc530ba (V32 pack),
5583c77b (MoE FP8), **BUG4 (this entry)**.

## What Worked

Widened ONLY the indexer gates to `mode.has_indexer()` (CompressedSparse +
SparseIndexed); left the COMPRESSOR gates on `has_compressor()`
(CompressedSparse/HybridCompressed). GLM has no key compressor — the indexer runs
over the FULL per-token latent (`compress_ratio` → 1, every token a key).

- **Gates widened to `has_indexer()`**: `dsa_official` / `dsa_key_cache` /
  `dsa_shared` instantiation + sizing; the indexer prepass + `csa_select` in
  both prepare paths (`mla_attention_prepare` prefill,
  `mla_attention_prepare_compressed_only` decode); the batched + single-row
  FlashMLA `selected_ptr` build sites; the decode forward's
  `use_batched_dsa_select` lane; the KV budget's shared-DSA + per-slot key-cache
  terms. Every SparseIndexed site substitutes `ratio=1` (the layer's nominal
  `compress_ratio` is 0 for GLM, which would panic `div_ceil` / `alloc(ratio*w)`).
- **Compressor gates untouched**: main `compressor_forward`, the
  `Dsv4CompressorState` indexer-compressor, `compressor_shape`, the opt-in
  compressor-batch prepass + full-flatten P1a all stay CompressedSparse/HCA-only.
- **New `sparse_indexed_index_key_forward`**: fills the index-KEY ring
  (`state.indexer.compressed`, bf16 `[index_head_dim, rows]`, ratio=1) from each
  token's hidden via the GLM `indexer.wk` projection + `k_norm` (vs the CSA path
  which derives keys through `indexer.compressor`). Confirmed against the vLLM
  DeepSeek-V3.2 indexer reference that `wk` is `Linear(hidden, index_head_dim)` —
  a SINGLE MQA key (128) shared across all `index_n_heads` query heads, NOT
  `n_head*head_dim`. GLM (`index_head_dim=128`) takes the implemented single-key
  branch; a wider `wk` fails loud (`bail!`) rather than fabricate a reduction.
- **MTP/full-flatten unsupported paths** (`commit_layer_fold`,
  `mla_attention_compressor_defer_row`) `bail!` loud for SparseIndexed — GLM
  ships no MTP (`num_nextn_predict_layers == 0`) and never engages opt-in
  full-flatten, so these are unreachable; failing loud beats silent garbage.

DSv4 CompressedSparse stays byte-identical: `index_ratio`/`idx_cr` evaluate to
exactly `compress_ratio` for CSA; only SparseIndexed maps to 1.

## Pending-remote (GPU validation)

Mac is CUDA-stubbed (no nvcc). Three `// ponytail: pod-verify`:
1. **GLM index `k_norm`** — the DSv3.2 reference normalizes the index key with
   `LayerNorm(index_head_dim, eps=1e-6)` (mean-subtract + weight + bias). GLM
   ships a `k_norm.bias`, implying LayerNorm. The current path applies bias-free
   `mla_rms_norm(config.rms_norm_eps)` — a correctness approximation to replace
   with a LayerNorm(+bias, eps=1e-6) kernel once a pod forward confirms GLM's
   exact index-key norm. `k_norm_bias` is kept live for that fix.
2. **`wk` index-key width** — expected `wk.rows == index_head_dim`; the `bail!`
   fires only on an unexpected checkpoint layout.
3. **SparseIndexed batched-decode DSA select lane** — end-to-end on the pod.

Verification on the pod: `scripts/needle_gate.py` correct-inference ladder on the
GLM-5.2 checkpoint (needle ×3 same-config vs envelope, NOT byte-identity) +
greedy-token decode of the generation to confirm the index norm path is correct.
No tok/s license needed for a correctness wire-up; a serving bench follows once
needle passes.

## Verify (Mac, all green)

- `CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib` — green
- `cargo clippy -p infer-cuda --release --no-default-features --features cuda,no-cuda --lib` — green (no new warnings)
- `cargo test -p deepseek-spec` — green (9 passed)

## Rule

Widen device-state/select gates by the CAPABILITY predicate (`has_indexer()`),
not the model-family enum variant (`== CompressedSparse`); substitute the
effective ratio (1 = no compression) at every indexer site rather than trusting
the layer's nominal `compress_ratio` (0 for a compressorless indexer). When a
checkpoint-specific numeric (index-key norm semantics) can't be GPU-verified,
wire the structure + a loud ponytail and an honest correctness-approximation
note — never silent garbage.
