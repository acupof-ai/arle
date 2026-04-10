# 2026-04-09 · Carnice-27b Q4_K_M OOM on L4-24GB during load

## Context

First attempt to load Carnice-27b (Qwen3_5ForCausalLM, 64L, hidden 5120, intermediate 17408) from `kai-os/Carnice-27b-GGUF` Q4_K_M (~16.5 GB on disk) onto a single L4-24GB. Server OOM'd around layer 16 of the weight upload loop. The original failure log was lost with the dev box — this entry is a reconstruction for the rule.

## Root Cause

`infer/src/weight_loader.rs::load_tensor_2d_gguf` treats any non-Q8_0 GGUF dtype as "dequantize to BF16 at load and upload BF16". The dispatch is:

```rust
if info.dtype == gguf::GgmlType::Q8_0 && info.shape.len() == 2 { /* packed fast path */ }
let bf16_data = gguf.read_tensor_bf16(&gguf_name)?;
DeviceMatrix::from_host(ctx, &bf16_data, rows, cols)
```

For Q4_K_M the actual on-GPU footprint is not 16.5 GB (the disk size) but ~54 GB (16-bit expansion), because each 4.5-bit-average superblock element becomes a full `bf16`. On 24 GB L4 the loop fills the GPU during the first ~60% of layers and allocates fail thereafter.

Secondary issue observed while reading the same code path: the V-head reorder helpers (`reverse_v_reorder`, `reverse_v_reorder_rows`, `reverse_v_reorder_cols`) do `let src = data.to_vec();` unconditionally — an extra full clone of the BF16 tensor per reordered weight. Peak CPU footprint per tensor ≈ 3× BF16 size (read_tensor_bf16 Vec + reorder src clone + from_host D2D staging). Not the OOM root cause, but a real bloat.

## Fix

See `docs/plans/q4k-native-gpu.md`. Summary:

- Keep Q4_K superblocks **packed** on the GPU; never materialise BF16.
- New CUDA kernel `q4k_gemv_kernel` reads 144-byte superblocks and dequant-fuses inside the GEMV.
- New rust path in `DeviceMatrix::from_quantized_q4k` and a Q4_K branch in `load_tensor_2d_gguf`.
- Linear-attention `out_proj` keeps the BF16 fallback (col reorder is not superblock-aligned and cannot be applied to packed bytes in place).

## Rule

- **Never assume "on-disk quantized size ≈ GPU resident size".** GGUF dequant-at-load silently triples to quadruples memory for Q4/Q3/Q5/Q6 schemes. Before shipping a new GGUF path, write down the expected GPU resident footprint per tensor class and compare against the physical GPU budget.
- **When adding a loader fast-path for one format (Q8_0), check whether the fallback for other formats is architecturally sound**, not just "correct at small model sizes". A dequant-at-load fallback that works for Qwen3-4B (8 GB BF16 on a 24 GB card) silently becomes a hard blocker for 27B-scale Q4_K models.
- **Don't rely on transient dev-machine logs for postmortems.** Write the error doc in the repo while the context is still in your head — the dev box will eventually die.
