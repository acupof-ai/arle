# NVFP4 checkpoint load + FP8 inference — DSv4-Flash-0731, CUDA TP=4, 2026-08-18

> Status: Shipped

## Goal

Serve the official DeepSeek-V4-Flash-0731 NVFP4 checkpoint (156 GB, 48 shards)
on 4×H20 with TP=4, using FP8 activation inference after load-time FP4→FP8
cache conversion. Delete the W4AFP8 (SGLang CUTLASS INT4) dead path that
shipped in earlier commits but never matched the checkpoint format.

## What changed

- **W4AFP8 deleted** — `QuantFormat::W4Afp8` variant, detection arms,
  `dsv4_moe_forward_w4afp8` (~256 lines), `W4Afp8ExpertWeights`, w4a8 FFI
  declarations + wrappers, `csrc/moe/w4a8/` kernel tree (6 files), CUTLASS 3.x
  build block, `quantize_dsv4_w4afp8.py`, `provision-cutlass-3x.sh`,
  `quantize-w4afp8` pod command.
- **NVFP4 loading added** — `DeviceMatrix::from_dsv4_fp4_block_scaled`
  constructor (packed E2M1 float4, 2 per byte, E8M0 per-1×32-block scales) +
  I8 arm in `load_dsv4_block_scaled`. The 0731 checkpoint's `.weight` (I8) +
  `.scale` (F8_E8M0) does not match any `detect_quant_format` arm, so
  `quant_view_for_dsv4` returns `None` and the FP8 MoE path routes here
  automatically. At load time `dsv4_block_scaled_to_fp8_deepgemm_cuda`
  converts FP4→FP8 E4M3 + FP32 per-128×128-block scales; inference uses the
  existing FP8 activation path.
- **compress_ratios 46→44** — 0731 ships 46 entries (43 hidden + 1 MTP + 2
  trailing). `from_json_value` truncates to `num_hidden_layers +
  num_nextn_predict_layers` before `validate()`.

## Parameters

```bash
bash scripts/pod.sh run nvfp4-v7 nvfp4-serve5 0,2,5,7 -- \
  serve --backend cuda \
  --model-path /data00/DeepSeek-V4-Flash-0731 \
  --tensor-parallel-size 4 --port 8000
```

- Baseline: none (first NVFP4 serve; W4AFP8 path was dead code, never ran)
- Treatment: HEAD (working tree) + `nvfp4-v7` build
- Trials: 1 (coherence + single-request decode)

## Environment

- Host / GPU: 8×H20 (sm_90, 96 GB), ranks on GPU 0,2,5,7
- Driver / CUDA: 12.8
- Model: DeepSeek-V4-Flash-0731 (NVFP4: routed experts I8 E2M1 + F8_E8M0,
  attention/shared expert FP8 E4M3 + E8M0, norms/embeddings BF16)
- TP / slots / KV: TP=4 / 59 slots (clamped from 256) / L1 GPU 0.9 + L2 host
  DRAM 50%
- Server flags: defaults (qwen35_decode_graph, batched_decode, deepgemm,
  moe_decode_kernel, gpu_router, fa3 all on)

## Results

| Metric | Value |
|--------|-------|
| Engine ready | yes (4/4 ranks) |
| VRAM per rank | 95 207 MB (weights 74 311 + adapter 2 297 + KV 18 727 − 130 residual) |
| Slots | 59 (clamped from 256) |
| Coherence (17×23) | 391 ✓ |
| Decode throughput (1 req, 64 tok) | ~51 tok/s |

Coherence check:

```
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -d '{"model":"...","messages":[{"role":"user","content":"What is 17 * 23? Give only the number."}],"max_tokens":16,"temperature":0}'
→ "391"
```

## Problems

- **Stale incremental CGU** — after deleting the w4a8 FFI, the `arle` link
  failed with `undefined symbol: w4a8_swiglu_fused` referenced by a cached
  codegen unit (`cuda_kernels.19aa83539241d259-cgu.0`). The `cuda-kernels`
  crate was recompiled but the stale CGU survived in `target/release/deps/`.
  Fix: `rm -rf target/release/deps/libcuda_kernels-* target/release/deps/cuda_kernels-*
  target/release/.fingerprint/cuda-kernels-* target/release/build/cuda-kernels-*`
  + rebuild.
- **Stale `deepseek-spec` rlib** — the compress_ratios truncate fix was in
  the source but cargo skipped recompiling `deepseek-spec` (tar-preserved
  mtime older than build). Fix: `rm -rf` the crate's artifacts + `touch
  crates/deepseek-spec/src/v4.rs` + rebuild.
- **`tn exec` vs container filesystem** — `tn exec` runs on the node;
  `/host/arle-build` is the container's view. Source verification via `tn exec`
  showed a stale/incomplete tree; `bin/pod` (in-container) confirmed the sync
  was correct.

## Learnings

PASS. NVFP4 checkpoint loads via FP4→FP8 cache conversion and serves with FP8
activations at TP=4 on H20. The W4AFP8 dead path is fully removed. Next wall:
needle-gate parity vs a reference baseline, then concurrency-sweep throughput
bench.
