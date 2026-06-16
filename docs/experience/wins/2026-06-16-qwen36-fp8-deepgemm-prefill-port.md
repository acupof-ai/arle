# Qwen3.6 FP8 prefill routed through DeepGEMM, correctness licensed

## SLO-shape probed? N

This run probed the bounded high-concurrency aggregate shape from the QAT plan:
512 input tokens, 32 output tokens, c=1..64. It did not include the full
4096/256 or 8K-prompt SLO shape, so this entry cannot license a default flip.

## Roofline check

Deferred. The point of this run was first to license correctness and remove the
measured >360s large-M prefill cliff. No nsys/ncu roofline trace was collected.

| Op | Achieved | Peak | % | Verdict |
|---|---:|---:|---:|---|
| Qwen3.6 FP8 routed MoE prefill | not measured | H20 peak | n/a | deferred: correctness and guidellm first |

## Goal

Port the existing native DeepGEMM FP8 grouped pipeline into the Qwen3.6 resident
quant MoE lane for prefill-size routed batches only, then re-run the FP8 vs BF16
aggregate c-sweep.

## Hypothesis

The hand FP8 grouped-GEMV path is decode-shaped and pathological for large-M
prefill. A DeepGEMM FP8 grouped prefill path should preserve the c<=256 routed
decode hand-kernel path, pass needle correctness, and improve aggregate
throughput once the serving front-door mutex cap is already fixed.

## Implementation

- Added FP8 grouped DeepGEMM caches for Qwen quant weights in
  `crates/infer-cuda/src/loader.rs`.
- Reused resident FP8 block-scaled tensors by copying them into contiguous
  `[expert, row, col]` grouped caches and pointer tables in
  `crates/cuda-kernels/src/tensor.rs`.
- Routed only large routed batches into DeepGEMM in `crates/infer-cuda/src/moe.rs`:
  `num_tokens * topk >= QWEN35_DEEPGEMM_MIN_ROUTES` (1024 routed rows).
  Decode remains on the hand quant kernel path.
- Ported the DSv4-style FP8 prefill sequence into Qwen MoE:
  pack/m_indices, BF16-to-FP8 quantize, FP8 grouped GEMM for fused gate/up,
  SwiGLU requantize, FP8 grouped down GEMM, scatter/combine.
- Added DeepGEMM JIT controls in
  `crates/cuda-kernels/csrc/gemm/deepgemm_native.cu` so runtime CUDA can stay on
  `/usr/local/cuda` while the JIT uses CUDA 12.9 plus `clang++-11`.

## Real checkpoint shape gate

Verified against
`/data01/models/Qwen3.6-35B-A3B-FP8/layers-0.safetensors`, not a comment:

| Tensor | Weight shape | Scale shape | K alignment |
|---|---:|---:|---:|
| gate | `[512, 2048]` | `[4, 16]` | 2048 % 128 = 0 |
| up | `[512, 2048]` | `[4, 16]` | 2048 % 128 = 0 |
| down | `[2048, 512]` | `[16, 4]` | 512 % 128 = 0 |

Config: hidden size 2048, MoE intermediate 512, 256 experts, top-k 8, 40 layers.

## Commands

Local verification:

```bash
cargo fmt -p infer-cuda -p cuda-kernels
cargo test -p infer-cuda --release --no-default-features --features no-cuda --lib
CUDARC_CUDA_VERSION=12060 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
CUDARC_CUDA_VERSION=12060 cargo clippy -p infer-api --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
git diff --check
```

Remote build:

```bash
CUDA_HOME=/usr/local/cuda \
CUDARC_CUDA_VERSION=12060 \
NVCC_CCBIN=/usr/bin/clang++-11 \
ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1 \
ARLE_CUDA_KERNEL_SET=dsv4_flash \
ARLE_CUDA_DISABLE_FLASHMLA=1 \
cargo build --release --features cuda
```

Remote serve:

```bash
CUDA_VISIBLE_DEVICES=0 \
INFER_TP_SIZE=1 \
ARLE_QWEN35_DEEPGEMM=1 \
CUDA_HOME=/usr/local/cuda \
CUDA_PATH=/usr/local/cuda \
ARLE_DEEPGEMM_JIT_CUDA_HOME=/usr/local/cuda-12.9 \
NVCC_CCBIN=/usr/bin/clang++-11 \
LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
target/release/arle serve \
  --backend cuda \
  --model-path /data01/models/Qwen3.6-35B-A3B-FP8 \
  --bind 127.0.0.1 \
  --port 8123 \
  --num-slots 999 \
  --total-pages 40 \
  --page-size 16 \
  --max-total-tokens 640 \
  --max-prompt-tokens 640
```

Guidellm sweep:

```bash
/root/dsv4-venv/bin/guidellm benchmark run \
  --target http://127.0.0.1:8123 \
  --model Qwen3.6-35B-A3B-FP8 \
  --processor /data01/models/Qwen3.6-35B-A3B-FP8 \
  --profile concurrent \
  --data prompt_tokens=512,prompt_tokens_stdev=1,prompt_tokens_min=512,prompt_tokens_max=512,output_tokens=32,output_tokens_stdev=1,output_tokens_min=32,output_tokens_max=32 \
  --max-seconds 60 \
  --backend openai_http \
  --backend-kwargs '{"validate_backend": "/v1/models", "request_format": "/v1/completions"}' \
  --disable-console-interactive \
  --outputs benchmark.json --outputs benchmark.csv \
  --rate 1,2,4,8,16,32,64 \
  --warmup 5
```

## Environment

- Backend: CUDA, H20 GPU0.
- FP8 model: `/data01/models/Qwen3.6-35B-A3B-FP8`.
- BF16 model: `/data01/models/Qwen3.6-35B-A3B`.
- Build feature set: `--release --features cuda`,
  `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1`.
- Runtime CUDA: `/usr/local/cuda` (12.5 library path).
- DeepGEMM JIT CUDA: `/usr/local/cuda-12.9`.
- DeepGEMM JIT host compiler: `/usr/bin/clang++-11`.
- Serve knobs: `--num-slots 999 --total-pages 40 --page-size 16
  --max-total-tokens 640 --max-prompt-tokens 640`.

## Correctness gate

Short greedy prompt decoded coherently:

```text
<think>

</think>

The capital of France is Paris.
```

Needle prompt used 243 prompt tokens and 256 generated tokens, so the routed
prefill crosses the DeepGEMM threshold (`243 * topk=8 > 1024`). It retrieved the
exact needle:

```text
Answer: ARLE-FP8-DEEPGEMM-NEEDLE-492817
```

Verdict: correctness PASS for the FP8 grouped DeepGEMM prefill lane.

## Results - slot and memory license

| Backend | Requested slots | Admitted slots | Peak VRAM MiB | Slot delta |
|---|---:|---:|---:|---:|
| FP8 DeepGEMM quant | 999 | 756 | 91,435 | +112.4% vs BF16 |
| BF16 baseline | 999 | 356 | 93,835 | baseline |

The memory/slot license still holds: the FP8 resident lane fits 756 slots in the
same H20 budget versus 356 for BF16.

## Results - c-sweep

Wall output tok/s is completed output tokens divided by the fixed 60s point
duration. The table uses the JSON files emitted by guidellm.

| c | FP8 ok/inc | FP8 wall out tok/s | FP8 TTFT ms | FP8 ITL ms | BF16 ok/inc | BF16 wall out tok/s | BF16 TTFT ms | BF16 ITL ms | FP8 delta tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12/1 | 6.98 | 2676.8 | 59.9 | 86/0 | 50.03 | 263.5 | 12.4 | -86.0% |
| 2 | 14/0 | 8.15 | 5321.8 | 112.4 | 110/2 | 64.00 | 448.1 | 17.2 | -87.3% |
| 4 | 12/4 | 6.98 | 10551.5 | 218.6 | 132/4 | 76.80 | 848.1 | 25.1 | -90.9% |
| 8 | 8/8 | 4.65 | 21733.0 | 563.0 | 143/8 | 83.20 | 1581.0 | 52.6 | -94.4% |
| 16 | 16/0 | 9.31 | 43223.3 | 1178.8 | 155/16 | 90.18 | 1973.7 | 132.4 | -89.7% |
| 32 | 0/32 | 0.00 | 0.0 | 0.0 | 153/32 | 89.02 | 5664.2 | 189.5 | -100.0% |
| 64 | 0/64 | 0.00 | 0.0 | 0.0 | 67/61 | 38.98 | 13939.0 | 720.1 | -100.0% |

Verdict: throughput FAIL. The port removes the prior non-returning prefill cliff
well enough to pass correctness and bounded requests, but FP8 still loses to BF16
at every completed c-point and completes no requests at c=32/64 within 60s.
This is not a default candidate and not "best kernel" evidence.

## Problems

- The first guidellm rerun completed workload execution but failed final save
  because `--outputs json --outputs csv` interacted badly with the output-dir
  alias. Re-running with explicit `benchmark.json` and `benchmark.csv` produced
  parseable artifacts.
- Building and running the whole server against CUDA 12.9 triggered a
  `libcublasLt.so.12` divide error during guidellm. Keeping runtime CUDA on
  12.5 and using CUDA 12.9 only for DeepGEMM JIT fixed the crash.
- CUDA 12.5 plus system `g++-8` could not compile the generated DeepGEMM C++20
  kernel. CUDA 12.9 plus `clang++-11` compiled the same generated kernel.

## Learnings

- Correctness is licensed, memory is licensed, throughput is not licensed.
- The next performance lever is not the HTTP front door and not a default flip.
  It needs nsys/ncu attribution inside the FP8 DeepGEMM path: input quantization,
  m-index packing, generated grouped GEMM selection, and SwiGLU requantization.
- Keep the decode hand kernel. This port intentionally only enters at 1024 routed
  rows, preserving the measured decode-side win.

## Artifacts

- FP8 JSON:
  `/data01/arle-qwenfp8-smoke/bench-output/qwen36-fp8-deepgemm-c1-64-512x32-jsonfix/benchmark.json`
- FP8 CSV:
  `/data01/arle-qwenfp8-smoke/bench-output/qwen36-fp8-deepgemm-c1-64-512x32-jsonfix/benchmark.csv`
- BF16 JSON:
  `/data01/arle-qwenfp8-smoke/bench-output/qwen36-bf16-c1-64-512x32-jsonfix/benchmark.json`
- BF16 CSV:
  `/data01/arle-qwenfp8-smoke/bench-output/qwen36-bf16-c1-64-512x32-jsonfix/benchmark.csv`

## Rule

Do not claim "best", "default", or "throughput win" from a port that only passes
correctness. Quant can ship as an opt-in memory/slot lever only if the runtime
surface makes that explicit; aggregate throughput still needs a separate
measured license.
