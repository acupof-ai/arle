# Qwen3.6 FP8 DeepGEMM SLO throughput gate: correctness and memory pass, throughput fail

## SLO-shape probed? Y

4096 input tokens / 256 output tokens, c=1,2,4,8, H20 GPU0. The guidellm
generator was set to `prompt_tokens=4095` because its 4096 setting produces
4097 tokens under the server tokenizer and aborts before entering the engine.
The emitted server usage for valid requests was exactly 4096 prompt tokens and
256 completion tokens.

## Goal

Run the final FP8-vs-BF16 throughput gate on the prefill-heavy SLO shape after
the Qwen3.6 FP8 resident quant lane was routed through native DeepGEMM for
large-R prefill.

## Hypothesis

If the FP8 DeepGEMM port is a throughput win, it should show up on the 4096/256
shape where prefill dominates and where the FP8 model's memory footprint permits
more slots. If it still loses there, H20 should be treated as compute-bound for
this QAT track: FP8 remains a memory/slot lever, not a raw throughput lever.

## Environment

- Remote tree: `/data01/arle-qwenfp8-smoke`.
- Binary: `/data01/arle-qwenfp8-smoke/target/release/arle`.
- Hardware: NVIDIA H20 GPU0.
- Runtime env: `ARLE_QWEN35_DEEPGEMM=1`,
  `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1` build,
  `CUDA_HOME=/usr/local/cuda`,
  `ARLE_DEEPGEMM_JIT_CUDA_HOME=/usr/local/cuda-12.9`,
  `NVCC_CCBIN=/usr/bin/clang++-11`.
- Serve shape: `--num-slots 999 --total-pages 272 --page-size 16
  --max-total-tokens 4352 --max-prompt-tokens 4096`.
- Workload: guidellm concurrent profile, `prompt_tokens=4095`,
  `output_tokens=256`, `--max-seconds 60`, `--warmup 5`.

## Slot and memory result

| Backend | Effective slots | Peak VRAM | Slot delta |
|---|---:|---:|---:|
| FP8 DeepGEMM quant | 381 | ~91.9 GiB | +111.7% vs BF16 |
| BF16 baseline | 180 | ~93.9 GiB | baseline |

The FP8 memory/slot license remains valid on the SLO shape.

## SLO throughput table

Metric is completed output tokens divided by the raw wall interval from first
request start to last request end in the guidellm JSON. `inc` counts requests
that were still incomplete at the fixed 60s point.

| c | FP8 ok/inc | FP8 out tok/s | FP8 TTFT ms | FP8 ITL ms | BF16 ok/inc | BF16 out tok/s | BF16 TTFT ms | BF16 ITL ms | FP8 delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2/1 | 6.54 | 20740.7 | 72.3 | 8/1 | 31.50 | 1760.9 | 24.9 | -79.3% |
| 2 | 0/2 | 0.00 | 0.0 | 0.0 | 8/2 | 33.28 | 3467.6 | 46.4 | -100.0% |
| 4 | 0/4 | 0.00 | 0.0 | 0.0 | 12/4 | 39.09 | 6844.6 | 75.9 | -100.0% |
| 8 | 0/8 | 0.00 | 0.0 | 0.0 | 8/8 | 33.93 | 13457.6 | 183.8 | -100.0% |

## Verdict

Outcome (b): FP8 still does not win on the SLO shape. The throughput gate is a
KILL for H20 raw throughput/default claims. This is now framed as H20 reality for
this track, not a new correctness bug: the FP8 lane is coherent and retrieves
the needle, native DeepGEMM is actually built, large-R prefill no longer hits the
old non-returning cliff, and the SLO run still loses to BF16.

Stop chasing raw FP8 throughput on H20 for QAT. Keep FP8 as an opt-in
memory/slot lever: it fits about 2.1x the long-shape slots in the same VRAM
budget, but BF16 remains the throughput baseline on this SKU.

## Artifacts

- FP8 JSON:
  `/data01/arle-qwenfp8-smoke/bench-output/qwen36-fp8-deepgemm-slo-4096x256-c1-8/benchmark.json`
- FP8 CSV:
  `/data01/arle-qwenfp8-smoke/bench-output/qwen36-fp8-deepgemm-slo-4096x256-c1-8/benchmark.csv`
- BF16 JSON:
  `/data01/arle-qwenfp8-smoke/bench-output/qwen36-bf16-slo-4096x256-c1-8/benchmark.json`
- BF16 CSV:
  `/data01/arle-qwenfp8-smoke/bench-output/qwen36-bf16-slo-4096x256-c1-8/benchmark.csv`

## Rule

When the synthetic bench generator and server tokenizer disagree by one token,
the apparent 4096-token run can be an ingress abort with zero engine steps. The
valid SLO gate is the server-usage shape, not the generator's nominal token
count.
