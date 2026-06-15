# Qwen3.6 CUDA Resident Quant Loader And MoE Dispatch

## SLO-shape probed?

Partial. This tranche wires Qwen3.6 FP8/NVFP4 resident quant weights into the
CUDA loader and routed/shared MoE decode dispatch. The remote gate now includes
a coherent FP8 serve smoke plus a same-binary BF16-vs-FP8 c=1, 512-in/32-out
guidellm A/B. It does not claim a throughput PASS or default flip; the resident
FP8 path only earns a memory license on this shape.

## Roofline check

Deferred. The routed MoE quant path intentionally uses the generic resident
GEMV correctness kernel first so FP8/NVFP4 checkpoints can load and run without
dense BF16 materialization. P8 must A/B this against DeepGEMM/CUTLASS/Marlin or
vendor kernels before any "best kernel" or default-performance claim.

Remote BF16 serve logs in the 2026-06-16 A/B also show DeepGEMM native MoE was
not available in this narrow build (`native_bridge=not_compiled`), so this run
is not an adopt-first kernel comparison. It is only BF16 resident path vs FP8
resident correctness-kernel comparison.

## Goal

Make Qwen3.6 CUDA quant checkpoints reachable through resident quant buffers:
attention and dense MLP projections load quant-aware, routed/shared MoE experts
preserve quant sidecars, and BF16-only decode fast paths stay fail-closed for
quant experts.

## What Worked

- Extended `SafetensorLoader` with cached safetensor headers, quant-manifest
  detection, and row/column/head-aware quant sharding for FP8 block-scaled,
  FP8 per-shard, FP4 E2M1 group, dense BF16, and dense F32 tensors.
- Switched Qwen3.6 attention q/k/v/o and dense MLP gate/up/down loaders to the
  quant-aware path. Follow-up remote serve reachability showed the official FP8
  checkpoint also quantizes `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, so
  those linear-attention projections now load through the same resident quant
  matrix path while embeddings, lm_head, routers, gates, norms, conv1d, and
  linear-attn small vectors stay dense.
- Routed MoE now records one expert `WeightFormat` plus scale/global pointer
  tables. Quant experts bypass the BF16 decode-fused and DeepGEMM paths and use
  the matching resident grouped GEMV wrapper.
- Added FP8 block-scaled and FP4 E2M1 grouped expert GEMV C ABI wrappers under
  the existing CUDA GEMM domain, keeping the kernel split aligned with the
  `cuda-kernels` module guide.
- Added `scripts/qwen36_dense_to_nvfp4.py` for offline dense Qwen3.6 ->
  ARLE-readable NVFP4 sidecar conversion. The script writes RedHat/unsloth ABI
  names (`weight_packed`, `weight_scale`, `weight_global_scale`,
  `input_global_scale`) and splits dense stacked experts into per-expert packed
  tensors for the current loader.

## Verification

```bash
CUDARC_CUDA_VERSION=12060 cargo check -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib
cargo test -p infer-cuda --release --no-default-features --features no-cuda --lib
cargo fmt -p infer-cuda -p cuda-kernels -- --check
CUDARC_CUDA_VERSION=12060 cargo clippy -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib -- -D warnings
python3 scripts/qwen36_dense_to_nvfp4.py --self-test
```

Results:

- `infer-api` CUDA/no-cuda typecheck: passed.
- `infer-cuda` no-cuda lib tests: 85 passed.
- `infer-cuda` + `cuda-kernels` fmt check: passed.
- `infer-api` CUDA/no-cuda clippy with `-D warnings`: passed.
- NVFP4 conversion script self-test, including float8 safetensors write/read:
  passed.

Remote gate status:

- CUDA feature compile on a clean remote sync passed with the narrow kernel-set
  build used for the FP8 serve smoke.
- Qwen3.6 FP8 serve smoke from `/data01/models/Qwen3.6-35B-A3B-FP8`: PASS.
  Decoded outputs were coherent:
  - Raw completion, known prompt: `<think></think> The capital of France is Paris.`
  - Raw completion, needle prompt: exact `ARLE-FP8-NEEDLE-738291`.
- Qwen3.6 FP8 perf gate, same binary A/B vs BF16 baseline: memory PASS,
  throughput FAIL on c=1, 512-in/32-out.
- NVFP4 serve smoke after generating or downloading a checkpoint. If HF download
  is slow, generate it from a dense Qwen3.6 checkpoint:

```bash
python3 scripts/qwen36_dense_to_nvfp4.py \
  --src /path/to/Qwen3.6-35B-A3B \
  --dst /path/to/Qwen3.6-35B-A3B-NVFP4-arle
```

- guidellm SLO sweep after the smoke path is correct.

Perf A/B command shape:

```bash
GUIDELLM_OUTPUTS="json csv" ./scripts/bench_guidellm.sh <label> \
  --target http://127.0.0.1:8123 \
  --model <Qwen3.6-35B-A3B or Qwen3.6-35B-A3B-FP8> \
  --processor </data01/models/...> \
  --concurrencies 1 \
  --data prompt_tokens=512,prompt_tokens_stdev=1,prompt_tokens_min=512,prompt_tokens_max=512,output_tokens=32,output_tokens_stdev=1,output_tokens_min=32,output_tokens_max=32 \
  --max-seconds 120 \
  --warmup 5
```

Environment: H20 GPU0, `CUDARC_CUDA_VERSION=12060`,
`ARLE_CUDA_KERNEL_SET=dsv4_flash`, `ARLE_CUDA_DISABLE_FLASHMLA=1`,
`INFER_TP_SIZE=1`, `--num-slots 1`, `--max-total-tokens 2048`,
`--max-prompt-tokens 2048`, commit `fe03c9b1` remote source snapshot. Peak VRAM
was sampled with `nvidia-smi` once per second. Tok/s below uses wall-clock
framing: completed output tokens / measured duration.

| Path | Completed / incomplete | TTFT mean | ITL mean | Output tok/s | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| BF16 | 37 / 1 | 2,760.64 ms | 12.38 ms | 10.30 | 69,455 MiB |
| FP8 resident | 4 / 1 | 26,410.29 ms | 59.92 ms | 1.11 | 37,423 MiB |
| FP8 delta vs BF16 | - | +856.67% | +384.14% | -89.19% | -46.12% |

License verdict: FP8 beats BF16 on peak VRAM for this binding shape, so memory
license PASS. FP8 does not beat BF16 on tok/s, TTFT, or ITL; throughput license
FAIL and the grouped quant-GEMV remains a correctness kernel pending the
adopt-first kernel A/B.

The first attempted canonical 4096-in/256-out c=1 run is invalid for perf:
`--max-seconds 60` expired before a request completed, so guidellm reported
`no successful requests recorded`. Do not use that run for a throughput verdict.

## Problems

- The grouped quant MoE kernel is a correctness/reachability kernel, not the
  final performance kernel. It deliberately avoids claiming "industry best"
  until a same-binary remote A/B measures the competing kernels.
- On the measured 512-in/32-out c=1 shape, FP8 is much slower than BF16 even
  though it uses much less memory. That is acceptable only as a memory license;
  it is a kill/iterate signal for throughput.
- The first FP8 serve attempt failed before readiness because the single-rank
  linear-attention loader still used BF16-only `load_matrix` for
  `linear_attn.in_proj_qkv.weight`, which the official FP8 checkpoint stores as
  `F8_E4M3`. That is a reachability gap, not a numeric verdict.
- NVFP4 checkpoint availability is remote-network dependent; use the conversion
  script above when HF download is slow, then run the serve/needle/guidellm gates
  on Colab or the CUDA host.

## Rule

Resident quant support is only useful if it never silently densifies full
weights. Loader support, sidecar sharding, and dispatch must fail closed on
unknown scale ABI, then earn performance with remote A/B before any default
claim.
