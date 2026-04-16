# 2026-04-10 · GGUF load path produces garbage forward output (unresolved)

## Context

While bringing up Carnice-27b Q4_K_M on L4-24GB (see
`2026-04-09-carnice-27b-q4k-oom.md`), the OOM was fixed by adding native
Q3_K / Q4_K / Q6_K packed GPU kernels and correcting the `GgmlType` enum
+ `dequant_q{3,4,5,6}_k` layouts to match llama.cpp. GPU weight residency
dropped from 66 GiB → 17.6 GiB (below llama.cpp's 17 GB baseline for 27B
Q4_K_M).

But forward-pass generation on **every** GGUF model we tested produces
stuck / incoherent tokens, regardless of Q-tier or model variant:

| Model                      | Output                          |
|----------------------------|---------------------------------|
| Carnice-27b Q4_K_M (GGUF)  | `"  A                    "`    |
| Qwen3.5-4B Q4_K_M (GGUF)   | `"backbackbackamer!!!"`         |
| Qwen3-4B   Q4_K_M (GGUF)   | `"!!!!!!!!"`                    |

Safetensors path for the same architectures passes the e2e tests. The
bug is exclusively in the GGUF load path.

## What's already ruled out

1. **Quantization kernels are correct.** `ground_truth_q4k.rs` pins
   Q4_K, Q5_K, Q6_K, and Q8_0 dequant against Python-computed llama.cpp
   ground truth for the first superblock of real Qwen3.5-4B tensors;
   all four match within bf16 precision.
2. **Native GPU packed kernels match CPU BF16 dequant** on real Carnice
   and Qwen3.5-4B tensors — `carnice_real_tensor_dequant.rs`
   (Q4_K ffn_gate, Q6_K ffn_down, Q4_K ssm_in_z with V-head row reorder)
   all agree within bf16 reduction noise.
3. **Enum values are canonical.** `GgmlType::{Q3_K=11, Q4_K=12, Q5_K=13,
   Q6_K=14, Q8_K=15, Q8_0=8, I8=24, I16=25, I32=26, BF16=30}`. Previous
   code had invented `Q4_K_S/M` = 14/15, `Q6_K` = 18 — disastrously
   wrong and meant no Q3/Q4/Q5/Q6_K tensor had ever been loaded
   correctly in this engine.
4. **`ssm_a` sign convention fixed.** Was `-log(|A|)`, should be
   `log(|A|)` per the `-exp(a_log) * softplus(...)` delta-rule kernel.
5. **RMSNorm offset subtraction** (`load_tensor_1d_gguf_offset_norm`
   subtracts 1.0 to match the `(1+w)*x` kernel) is self-consistent; toggling
   it via `INFER_DEBUG_NO_NORM_SUB` changes nothing.
6. **Forcing BF16 dequant for all 2D tensors** via
   `INFER_FORCE_BF16_QUANT=1` still produces garbage — just a
   *different* garbage. Rules out the native GPU packed path as the
   root cause: the shared CPU dequant + upload path is equally broken.
7. **Linear attention is not the cause.** Qwen3-4B (pure dense
   transformer, no linear attention) also breaks under GGUF load.

## Update (2026-04-10 night) — layer-by-layer instrumentation

Added `debug_dump_hidden()` in `model::common` (gated by
`INFER_DEBUG_DUMP=1`) and wired it through the Qwen3 prefill loop to
dump head[0..8], tail[N-4..N], min/max/rms at four checkpoints per layer.
Side-by-side run of Qwen3-4B safetensors vs Q4_K_M GGUF on the prompt
"The capital of France is" yields:

| Layer | safetensors rms | GGUF rms (bf16-forced) |
|------:|----------------:|-----------------------:|
|  L0   | 0.151           | 0.120                  |
|  L1   | 0.189           | 0.154                  |
|  L2   | 0.224           | 0.177                  |
|  L3   | 0.285           | 0.217                  |
|  L4   | 0.383           | **0.420** (first cross-over) |
|  L5   | 0.501           | **16.06** ← 32× explosion in ONE layer |
|  L6   | 0.560           | 21.30                  |
|  L11  | 0.843           | 223.9                  |

L0..L4 drift is 22-30% per layer — exactly what 20% per-tensor RMS
quantization noise would compound to. L5 explodes 38× in a single
layer, then cascades.

Cross-validations done:
- `llama-cpp-python` on the SAME GGUF file produces correct text
  (" Paris. The capital of Germany is Berlin. The capital of"). The
  weights are fine; our forward pass is what's broken.
- `blk.5.input_layernorm.weight` matches BIT-EXACT between safetensors
  and GGUF (both F32 → BF16 in the loader; GGUF stores F32 verbatim).
- `blk.5.attn_q.weight` row 0 / row 1 dequant matches safetensors
  within Q4_K precision (5-15% per element).
- The 20% systematic per-tensor RMS inflation is REAL Q4_K
  quantization behavior — Python ground-truth dequant matches our
  Rust dequant matches llama.cpp's `dequantize_row_q4_K` byte-for-byte.

Per-layer dtype audit shows the Q4_K_M recipe interleaves Q6_K and Q4_K
for `attn_v` / `ffn_down`: blk.{0,1,2,3,6,9,12,15,18,21,24,27,30..35} use
Q6_K, the rest use Q4_K. Layer 5 uses Q4_K — and is exactly where
explosion starts. But layer 4 ALSO uses Q4_K and only drifts ~10%, so
"Q4_K vs Q6_K" alone doesn't explain it.

So the bug is now definitively in **the Qwen3 forward pass code path**,
specifically: it tolerates 20% per-tensor RMS quant noise much worse
than llama.cpp does. Likely candidates:
- A specific kernel using bf16 storage between fp32 stages where
  llama.cpp keeps fp32 (silu_mul fixed already as a real-but-not-the-bug
  issue, found while reading the kernel).
- Attention softmax overflow / stability — flashinfer should be stable
  but worth verifying it max-subtracts before exp.
- Accumulator dtype in attention or RoPE.

## Update (2026-04-10 evening)

One more real bug found and fixed: **`embed_tokens` was being uploaded
as a packed-quantized `DeviceMatrix` with a 1-element dummy `.data`
buffer**, but `embedding_decode_cuda` reads from `embed.data` directly
(not quant-aware). Every forward pass started by reading garbage
device memory through the embedding lookup. Fixed by adding
`load_tensor_2d_gguf_bf16` that forces BF16 dequant, and using it for
`embed_tokens` in both Qwen3 and Qwen3.5 GGUF loaders.

Differential: Qwen3-4B GGUF output changed from `"!!!!!!!!"` (reading
dummy memory) to prompt-dependent tokens like `"零食零食零食..."` —
the model is now processing input through its embedding correctly,
but still stuck in repetitive loops.

Additional verification completed:
- `blk.0.input_layernorm.weight[0..8]` matches exactly between
  safetensors (BF16) and GGUF (F32 → BF16).
- `model.norm.weight[0..8]` (final norm) also matches exactly.
- `blk.0.attn_q.weight[row 0, 0..8]` and `[row 1, 0..8]` both match
  safetensors within Q4_K quantization noise (~5-15%). Row stride is
  correct.
- Existing safetensors-only e2e test on Qwen3-4B produces COHERENT
  English generation (just slight baseline drift from the hard-coded
  expected string). The forward pass with safetensors weights works.

So the weight data on device is numerically correct vs safetensors,
and the forward pass is correct with safetensors weights, yet the
forward pass with GGUF weights (same numerical values within quant
noise) produces stuck tokens. This strongly suggests a subtle issue
that is NOT in weight loading or kernel math but rather in some
shape / DeviceMatrix metadata set-up that differs between the two
load paths (e.g., a stride, an alignment, or a `rows`/`cols` field
accessed by a later op).

## What's left to investigate

The bug lives somewhere in the shared Qwen3/Qwen3.5 **GGUF loader**
code that is NOT exercised by the safetensors path and NOT covered by
our ground-truth tests:

- **Tensor name mapping**: `map_gguf_name_with_prefix` in `gguf.rs`
  (blk.X → layers.X.*). A wrong mapping would silently load the wrong
  tensor into a slot whose shape happens to match, e.g., mapping
  `attn_q` → `mlp.gate_proj.weight` for a model where those shapes
  coincide.
- **Row/column interpretation for 2D tensors**: `load_tensor_2d_gguf`
  returns `(rows=ne1, cols=ne0)`. The byte layout cross-check
  (`ground_truth_q4k.rs`) only verifies the *first superblock of row 0*
  — it does not verify that rows 1..N are in the right order. A
  transposed row stride would produce correct data for row 0 but
  permuted for every other row.
- **Embedding lookup**: `token_embd.weight` dtype is Q6_K for
  Qwen3-4B (shape `[2560, 151936]`). The embedding kernel reads
  `embed[token_id, :]`; if the row layout after dequant doesn't match
  what the kernel expects, every token gets the wrong hidden state
  from the very first step.
- **Attention `q_proj` storage order for Qwen3.5**: for full attention
  with `attn_output_gate=true`, `attn_q.weight` has 2× output dim.
  Spot-checking RMS of per-256-row stripes of `blk.3.attn_q.weight`
  showed an alternating pattern consistent with interleaved
  `[h0.q(256), h0.gate(256), h1.q(256), ...]` — which matches
  `decode_prep_paged_hd256_kernel`'s expectation. So this likely isn't
  the bug, but it's the sort of thing that was not previously asserted.

## Next debug step

Build a single-layer diagnostic test that loads **one** Qwen3-4B GGUF
and compares hidden states at each layer boundary against the same
model loaded from safetensors. The first layer where they diverge is
the bug location. This needs a Qwen3-4B safetensors checkpoint locally
(not currently on disk) — download from `Qwen/Qwen3-4B`.

## Rule

- **"Load succeeds + tensor cross-checks pass" ≠ "model works".**
  Quantization correctness and shape-matching at load time tell you the
  bytes on device are the right bytes; they tell you nothing about
  whether those bytes are pointed at by the right name / are in the
  expected row order / get preprocessed correctly. Always force a
  single-forward diagnostic that compares two load paths end-to-end
  before declaring a loader correct.
- **Ground-truth tests should check more than row 0.** Verifying
  element `[0, 0]` and `[0, 31]` of a weight tensor catches dequant
  math bugs but misses row-stride bugs. Always include a sample from a
  late row (e.g. `[rows-1, 0]`) so transposition and stride errors
  surface.
- **Keep a `*_FORCE_BF16_QUANT` env-var bisect hook** in the loader
  path. The ability to bypass the native GPU packed fast-paths and go
  through CPU dequant is essential for telling "kernel bug" apart from
  "layout / naming bug" — without it, you're guessing.
