# Q4_K Native GPU Kernel — Carnice-27b on L4-24GB

> **Status: Shipped** (post 2026-04-10). This plan describes the design that
> actually landed. The native Q4_K path is in production:
>
> - Kernel: [`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`](../../crates/cuda-kernels/csrc/gemm/quantized_gemv.cu)
>   (`q4k_gemv_kernel` + batched variant)
> - FFI: [`crates/cuda-kernels/src/ffi/gemm.rs`](../../crates/cuda-kernels/src/ffi/gemm.rs)
>   (`q4k_gemv_cuda`)
> - `DeviceMatrix::from_quantized_q4k`: [`crates/cuda-kernels/src/tensor.rs`](../../crates/cuda-kernels/src/tensor.rs)
> - GGUF packed reader: `gguf::read_tensor_q4k_packed` in [`infer/src/gguf.rs`](../../infer/src/gguf.rs)
> - Loader fast path: [`infer/src/weight_loader.rs`](../../infer/src/weight_loader.rs)
>   (`load_tensor_2d_gguf` Q4_K branch at ~line 879, `load_tensor_2d_gguf_v_reorder_rows`
>   at ~line 722)
> - Dispatch: [`infer/src/ops/linear.rs`](../../infer/src/ops/linear.rs) — `quant_bits == 44` → q4k path
> - Tests: [`infer/tests/q4k_kernel_correctness.rs`](../../infer/tests/q4k_kernel_correctness.rs)
>   (419 lines), [`infer/tests/ground_truth_q4k.rs`](../../infer/tests/ground_truth_q4k.rs),
>   [`infer/tests/smoke_carnice_27b_q4k.rs`](../../infer/tests/smoke_carnice_27b_q4k.rs)
>
> Related bug-resolution experience entries:
> [`docs/experience/errors/2026-04-09-carnice-27b-q4k-oom.md`](../experience/errors/2026-04-09-carnice-27b-q4k-oom.md),
> [`docs/experience/errors/2026-04-10-gguf-load-path-forward-garbage.md`](../experience/errors/2026-04-10-gguf-load-path-forward-garbage.md),
> [`docs/experience/errors/2026-04-10-remaining-gguf-bugs.md`](../experience/errors/2026-04-10-remaining-gguf-bugs.md).
>
> The design below is preserved as a reference for the rationale (tensor
> coverage decision, superblock alignment argument, kernel layout). The
> "Next step" section at the bottom is historical — steps 1→8 have all
> been implemented.

## Goal

Load Carnice-27b (Qwen3_5ForCausalLM, 64L, 5120 hidden, 17408 intermediate) from a Q4_K_M GGUF (~16.5 GB on disk) onto a single L4-24GB, with **GPU resident memory matching the quantized size expectation** (~16-18 GB including the BF16 fallback subset + KV/activations).

## Why current path fails

`load_tensor_2d_gguf` (weight_loader.rs) dequantizes non-Q8_0 tensors to BF16 at load time via `gguf.read_tensor_bf16()`, then `DeviceMatrix::from_host` uploads them to the GPU. For Carnice-27b Q4_K_M:

- Disk: 16.5 GB (4.5 bits/element avg)
- BF16 expansion: ~54 GB (16 bits/element)
- **L4-24GB OOM around layer 16** — recorded in the now-lost `2026-04-09-carnice-27b-q4k-oom.md` report.

There is also a transient double-copy on the CPU side:
1. `gguf.read_tensor_bf16()` materialises a full `Vec<bf16>` of the tensor.
2. `reverse_v_reorder_*` helpers do `let src = data.to_vec();` — **clones the entire BF16 tensor again** as a scratch buffer for out-of-place permutation.
3. `DeviceMatrix::from_host` then clones it a third time to the GPU.
Peak CPU footprint per tensor ≈ 3× BF16 size, but this is dominated by the real killer (GPU side).

## Strategy — keep Q4_K packed on GPU

Add a native Q4_K `DeviceMatrix` variant that uploads the raw 144-byte-per-superblock GGUF bytes verbatim. Write a Q4_K-specific CUDA GEMV (decode) and a chunked dequant-to-tile + cuBLAS path for prefill.

### Tensor coverage decision

Q4_K superblocks span 256 contiguous K-dimension elements. Any reorder that is **row-aligned** (permuting output features) can be applied as whole-row byte copies — superblock integrity preserved. **Column reorders** that rearrange within K at a granularity that is not a multiple of 256 break superblock structure.

Carnice-27b linear attention has `num_k_heads=16`, `num_v_heads=48`, `head_dim=128`. The V-head reorder used by llama.cpp permutes columns at 128-element granularity → **NOT superblock-aligned**.

| Tensor class | Reorder | Q4_K path |
|---|---|---|
| `embed_tokens`, `lm_head` | none | **Q4_K packed** |
| Full-attention `q_proj / k_proj / v_proj / o_proj` | none | **Q4_K packed** |
| Linear-attention `in_proj_qkv` | none | **Q4_K packed** |
| Linear-attention `in_proj_z / a / b` | row (V-head along rows) | **Q4_K packed** + byte-level row permute on upload |
| Linear-attention `out_proj` | col (V-head along cols) | BF16 fallback (existing path) |
| MLP `gate_proj / up_proj / down_proj` | none | **Q4_K packed** |
| `conv1d_weight`, all 1D norms / biases | varies | BF16 (size negligible) |

Estimated GPU footprint:

- Q4_K packed (majority): ~15.5 GB
- BF16 out_proj fallback: 48 layers × (5120 × 6144 × 2 bytes) ≈ 3.0 GB
- 1D weights, norms, conv1d: <200 MB
- Activation / KV cache per slot: ~1-2 GB
- **Total target: ~19-21 GB** on 24 GB L4.

## Data layout (GPU-resident)

Per row (output feature) of a 2D weight `[rows=N, cols=K]` with K % 256 == 0:

```
row r: [ SB_0 | SB_1 | ... | SB_{K/256 - 1} ]
SB_i:  144 bytes  =  d:f16 | dmin:f16 | scales_packed:[12 B] | qs:[128 B]
```

This is verbatim the GGUF on-disk layout (GGUF stores row-major once reinterpreted from its ne0/ne1 order — same reinterpretation as the existing Q8_0 fast path at weight_loader.rs:775). Per-row byte stride = `(K / 256) * 144 = K * 0.5625`.

Upload path: `mmap` → slice out the per-tensor byte range → `clone_htod` straight to GPU as `CudaSlice<u8>`. **No BF16 intermediate ever materialises.**

## CUDA kernel — `q4k_gemv_kernel`

**Decode (M=1) GEMV.** Block layout mirrors existing `w4a16_gemv_kernel`:
- 8 rows × 32 threads = 256-thread block
- Each row processed by 32 threads (one warp).
- Each warp iterates over all `K/256` superblocks in its assigned row, striding by warp lanes at the sub-block (32-element) granularity → 8 sub-blocks/superblock / 32 lanes = lanes cover the 8 sub-blocks in 1 pass with 24 lanes idle, OR distribute at 16-byte (32-nibble) chunks within the 128-byte qs region.

Revised thread→data mapping (simpler, matches existing kernel style):
- 1 warp per row.
- Loop `sb = 0 .. K/256`:
  - Lane 0-7 cooperate to load (d, dmin) once into shared memory (1 broadcast).
  - All 32 lanes read `scales[12]` and decode 8 sub-scales + 8 sub-mins in registers (broadcast-safe).
  - Each of the 8 sub-blocks (j = 0..8): 32 lanes process the 32 nibbles (16 bytes). Lane `l` handles nibble index `l`: byte `l/2`, low/high based on `l%2`.
  - Dequantized value: `w = d * sub_scale[j] * q - dmin * sub_min[j]` where q is the raw nibble (unsigned, 0..15).
  - Multiply by `input[sb*256 + j*32 + l]`, accumulate into thread-local `sum`.
- Warp reduce `sum`, lane 0 writes `output[row]`.

Key points:
- 128-bit vector load (`uint4`) for the 16 bytes of each sub-block's qs — 1 transaction/sub-block.
- `d` and `dmin` are fp16 on disk; cast to float once per superblock.
- Scales decode: 12-byte block is small enough to live in shared memory per warp, decoded once.
- Input activations for a superblock span 256 contiguous elements → `reinterpret_cast<const uint4*>` vectorised loads match coalescing principle #1.

**Batched prefill (M>1).** Use a chunked dequant-to-tile strategy rather than a native Q4_K tensor-core GEMM (deferred to a follow-up):
- Allocate persistent `[N, chunk_K]` BF16 workspace (chunk_K = 4096, size per row = 8 KB, total 40-200 MB depending on N).
- For each K-chunk: launch `q4k_dequant_chunk_kernel` that writes the BF16 tile; call existing `gemm_cuda` (cuBLAS) on the tile; accumulate into output.
- This reuses cuBLAS tensor cores and is bandwidth-bound on dequant. Good enough as a correctness baseline; a dedicated Q4_K Marlin variant can follow.

## Rust-side changes

1. **`gguf.rs`** — add `read_tensor_q4k_packed(name) -> Result<Vec<u8>>` (thin wrapper around `read_tensor_raw`, validates dtype + block count).
2. **`tensor.rs`** — add `DeviceMatrix::from_quantized_q4k(ctx, packed_bytes, rows, cols)`. New field: we will overload `quant_bits` — use `quant_bits = 44` as the Q4_K discriminator (4-bit, K superblock) to avoid changing the struct shape. The existing W4A16 path continues to use `quant_bits=4`. `group_size` is set to 256 (superblock size) as informational only.
3. **`ffi.rs`** — extern bindings: `q4k_gemv_cuda`, `q4k_gemv_batch_cuda`, `q4k_dequant_chunk_cuda`.
4. **`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`** — add `q4k_gemv_kernel` + batched variant + C wrappers.
5. **`crates/cuda-kernels/csrc/quant/q4k_dequant.cu`** (new) — `q4k_dequant_chunk_kernel` for prefill tile, plus wrapper.
6. **`ops/linear.rs`** — dispatch `quant_bits == 44` to `q4k_gemv_cuda` (decode) and to the dequant-chunk + cuBLAS path (prefill).
7. **`weight_loader.rs`**:
    - Add Q4_K fast path to `load_tensor_2d_gguf` (mirror the Q8_0 branch).
    - Add `load_tensor_2d_gguf_q4k_v_reorder_rows` that permutes rows at byte-stride granularity (`row_bytes = K * 9 / 16`).
    - Keep existing `load_tensor_2d_gguf_v_reorder_cols` unchanged — it remains BF16 for `out_proj`.
8. **`model/qwen35/weights.rs`** — no structural change; only the loader dispatch picks the Q4_K path transparently.
9. **`model/qwen35/config.rs`** — already supports `Flat(TextConfig)` for Carnice-27b. Verify the 27B-specific head_dim=256 / num_heads=24 works with existing FlashInfer HD256 path.

## Verification plan

1. **Synthetic unit test**: build a small `[rows=16, cols=512]` Q4_K tensor (2 superblocks/row), dequant on CPU (existing `dequant_q4_k`), upload to GPU via both paths, run GEMV, compare `y` ≤ 1e-2 relative error.
2. **Q4_K_M on Qwen3.5-4B** (if a Q4_K_M GGUF exists for it): run e2e regression against the BF16 baseline in `infer/test_data/`. Max token-level logits delta should be small (Q4_K inherently introduces quantization noise vs. BF16 reference — define tolerance at first observation).
3. **Full load test**: load Carnice-27b, measure `cudaMemGetInfo` before/after load, assert resident ≤ 21 GB.
4. **Generation smoke test**: prompt "The capital of France is", greedy decode 16 tokens, verify sensible completion (no NaN / garbage).
5. **Memory snapshot** → `docs/experience/wins/YYYY-MM-DD-carnice-27b-q4k-native.md` with before/after GPU footprint and tok/s.

## Out of scope (follow-up)

- Native Q4_K tensor-core GEMM (Marlin-style) — current plan uses chunked dequant + cuBLAS.
- Q5_K / Q6_K native kernels.
- Fixing the linear-attention col-reorder case for packed weights (requires dequant + requant, or custom kernel with an in-K permutation LUT).

## Next step (historical)

Implement 1→8, then run verification 1→5. **All eight implementation steps
and verification 1–5 have landed** — see the status banner at the top of
this file for the final code locations.
