# Metal Qwen3.5 0.8B GGUF Q4_K_M decode >200 tok/s

## Goal

- Bring the Metal GGUF Qwen3.5-0.8B Q4_K_M decode hot path onto MLX affine/tiled kernels and cross 200 tok/s for the 512/1024 decode target.

## Hypothesis

- Q4_K_M still had large Q5_K and Q8_0 tensors outside the affine path, and the tied Q6_K lm_head needed a larger MLX qmv tile. Repacking Q5_K/Q8_0 plus a Q6/group16 qmv tile change should remove the remaining raw/scalar bottleneck.

## Command

```bash
cargo build -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench

AGENT_INFER_QWEN35_GENERATE_PROFILE=1 ./target/release/metal_bench \
  --model /path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 512 \
  --generation-tokens 1024 \
  --warmup 0 \
  --runs 1 \
  --ignore-eos \
  --json

AGENT_INFER_QWEN35_GENERATE_PROFILE=1 ./target/release/metal_bench \
  --model /path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 32 \
  --generation-tokens 256 \
  --warmup 1 \
  --runs 1 \
  --ignore-eos \
  --json
```

## Environment

- Backend: Metal
- Model: `/path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf`
- Hardware: Apple M4 Pro, 20 GPU cores, 48 GB unified memory, Metal 4
- OS: Darwin 25.3.0 arm64
- Commit before change: `64205c7`
- Feature set: `cargo build -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench`
- Non-default bench env: `AGENT_INFER_QWEN35_GENERATE_PROFILE=1`

## Results

| profile | generation tok/s | repo e2e tok/s | decode avg ms/token | prompt tok/s | TTFT ms | peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|
| 512 prompt / 1024 decode | 211.719 | 202.367 | 4.723 | 2290.576 | 223.525 | 1426.094 |
| 32 prompt / 256 decode | 227.298 | 224.515 | 4.400 | 2292.359 | 13.959 | 1424.484 |

Raw 512/1024 profile:

```text
decode tokens=1024 total=4836.598ms avg=4.723ms forward_build_avg=0.480ms sample_build_avg=0.002ms async_avg=4.152ms eval_wait_avg=0.000ms item_avg=0.000ms bookkeep_avg=0.085ms clear_cache_total=11.207ms last_intermediates=582
```

Raw 32/256 profile:

```text
decode tokens=256 total=1126.277ms avg=4.400ms forward_build_avg=0.441ms sample_build_avg=0.001ms async_avg=3.889ms eval_wait_avg=0.000ms item_avg=0.000ms bookkeep_avg=0.061ms clear_cache_total=1.148ms last_intermediates=582
```

## Delta vs baseline

- Baseline: `docs/experience/wins/2026-04-27-bench-metal-qwen35-0p8b-gguf-affine-repack.md`

| profile | baseline tok/s | now tok/s | delta |
|---|---:|---:|---:|
| 512 prompt / 1024 decode | 82.521 | 211.719 | +156.6% |
| 32 prompt / 256 decode | 87.580 | 227.298 | +159.5% |

## What Changed

- Repacked GGUF Q5_K weights into MLX affine group-32 layout.
- Repacked GGUF Q8_0 weights into MLX affine group-32 layout by mapping signed int8 to unsigned affine form (`scale=d`, `bias=-128*d`).
- Tuned vendored MLX Metal `qmv_fast` for Q6/group16 by using 4 packs per thread and 4 SIMD groups per threadgroup, with matching host-side `bn=16` dispatch.

## Problems

- A direct fused lm_head+argmax experiment was slower than MLX qmv and was removed.
- Repacking grouped-value `out_proj` columns into affine was also slower than the existing raw packed input-reorder path and was removed.
- `cargo test -p infer --release --no-default-features --features metal,no-cuda ...` is currently blocked by unrelated dirty `infer/src/server_engine.rs` changes: one `CompletionRequest` test initializer is missing `session_id` and `trace_context`.

## Learnings

- For this GGUF, the tied `token_embd`/lm_head is Q6_K and dominates decode after Q5_K GDR matrices are repacked.
- The useful lm_head fix is not a bespoke argmax kernel; it is aligning MLX's Q6/group16 qmv tile and host dispatch with the very large vocab projection.
