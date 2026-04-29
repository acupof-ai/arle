# SGLang Alignment c=16 Bench - CUDA L4, 2026-04-29

## Goal

- Align ARLE's operator-facing CUDA serving knobs with SGLang defaults and run
  same-host c=16 GuideLLM comparisons for Qwen3-4B and Qwen3.5-4B.
- Goal type: regression + competitor parity.

## Hypothesis

- SGLang should have lower TTFT from its scheduler/process split and prefill
  path, while ARLE should remain competitive on decode ITL after graph warmup.
- Short prompts should not pay prefix prefetch / PD split overhead; they should
  run as local cold prefill and avoid `decode+prefill` split launches.

## Params

- Workload: GuideLLM `concurrent`, c=16, 4096 input tokens, 256 output tokens,
  120 seconds, seed `20260416`.
- Backend: OpenAI HTTP `/v1/completions`.
- GPU: NVIDIA L4, 23034 MiB, driver 580.82.07.
- SGLang: `0.5.10.post1`, torch `2.9.1+cu128`.
- ARLE build:
  `CARGO_HOME=/tmp/arle-cargo-home CARGO_TARGET_DIR=/tmp/arle-target ZIG=/tmp/zig-0.15.2/zig-x86_64-linux-0.15.2/zig CUDA_HOME=/usr/local/cuda cargo build --release -p infer --bin infer --features cuda`
- Serial rule: one server on the single GPU at a time. Ran SGLang first, then
  ARLE for each model.

## Env

- Raw artefacts:
  - `bench-output/2026-04-29-sglang-cuda-l4-qwen3-c16-sglang-align-run2/`
  - `bench-output/2026-04-29-arle-cuda-l4-qwen3-c16-sglang-align/`
  - `bench-output/2026-04-29-sglang-cuda-l4-qwen35-c16-sglang-align/`
  - `bench-output/2026-04-29-arle-cuda-l4-qwen35-c16-sglang-align/`
- GPU samples:
  - `bench-output/gpu-samples/2026-04-29-sglang-qwen3-c16-run2-dmon.txt`
  - `bench-output/gpu-samples/2026-04-29-arle-qwen3-c16-sglang-align-dmon.txt`
  - `bench-output/gpu-samples/2026-04-29-sglang-qwen35-c16-sglang-align-dmon.txt`
  - `bench-output/gpu-samples/2026-04-29-arle-qwen35-c16-sglang-align-dmon.txt`

## Parameter Alignment

| Surface | SGLang observed/default | ARLE status after this change |
|---|---|---|
| chunked prefill | `--chunked-prefill-size`; bench pinned `512` | Existing CLI and scheduler; bench pinned `512`. Auto HBM table still differs on L4 (`2048`), so explicit flag remains needed for same-condition bench. |
| max prefill tokens | default/bench `16384` | Existing CLI/config, default `16384`. |
| schedule policy | default `fcfs` | Added `--schedule-policy fcfs`; unsupported policies fail fast instead of becoming no-ops. |
| radix/prefix cache | enabled by default, `--disable-radix-cache` | Added `--disable-radix-cache`; default on. |
| KV cache dtype | `--kv-cache-dtype auto/fp8_e4m3/...`; bench used `fp8_e4m3` | Existing `--kv-cache-dtype fp8`; ARLE stores paged KV as FP8E4M3 with BF16 contiguous prefill. |
| CUDA graph | enabled by default, `--disable-cuda-graph` | Added `--disable-cuda-graph` alias in addition to `--cuda-graph=false`. |
| piecewise CUDA graph | SGLang Qwen3: enabled; SGLang Qwen3.5: disabled automatically by model/Mamba path | ARLE Qwen3 decode graph capture exists; ARLE Qwen3.5 piecewise decode graph capture stays enabled. |
| torch.compile | SGLang exposes `--enable-torch-compile`, default off in observed args | Not applicable in Rust hot path. |
| stream interval | default `1` | Added `--stream-interval`, default `1`, emit worker buffers by generated-token interval. |
| attention backend | SGLang `--attention-backend flashinfer` in bench | ARLE CUDA uses FlashInfer/TileLang internally; no user-facing backend selector yet. |
| tokenizer/detokenizer split | SGLang tokenizer/detokenizer/scheduler processes | ARLE already has emit/detokenization worker; broader tokenizer/scheduler overlap remains profiling-backed follow-up. |
| overlap scheduler | SGLang Qwen3 enabled; SGLang Qwen3.5 disabled by mamba `no_buffer` | ARLE loop overlaps readback across ticks and emit worker; no full scheduler/model-worker process split yet. |
| short prompt bypass | SGLang has chunked prefix-cache bypass controls | Added `--short-prompt-bypass-tokens` default `256`; short prompts skip prefix lookup/staged prefetch/publish and avoid legacy `decode+prefill` split launches. |

SGLang references used for the mapping: official server arguments docs and
`sglang/srt/server_args.py` defaults. Local observed values came from
`server_args=ServerArgs(...)` logs for the exact commands below.

## Commands

```bash
python3 -m sglang.launch_server \
  --model-path infer/models/Qwen3-4B \
  --served-model-name Qwen/Qwen3-4B \
  --host 127.0.0.1 --port 8000 \
  --mem-fraction-static 0.85 \
  --max-running-requests 16 \
  --chunked-prefill-size 512 \
  --max-prefill-tokens 16384 \
  --schedule-policy fcfs \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend flashinfer \
  --stream-interval 1

/tmp/arle-target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --chunked-prefill-size 512 \
  --max-prefill-tokens 16384 \
  --prefill-max-requests 16 \
  --kv-cache-dtype fp8 \
  --mem-fraction-static 0.85 \
  --schedule-policy fcfs \
  --stream-interval 1

scripts/bench_guidellm.sh <label> \
  --target http://127.0.0.1:8000 \
  --model <served-model> \
  --processor <local-model-path> \
  --concurrencies 16 \
  --max-seconds 120
```

Qwen3.5 used the same shape with model `Qwen/Qwen3.5-4B` and
`--mem-fraction-static 0.70`.

## Results

| Model | Server | out tok/s | TTFT p50 ms | ITL p50 ms | E2E mean s | Notes |
|---|---:|---:|---:|---:|---:|---|
| Qwen3-4B | SGLang | 133.74 | 8375.7 | 86.20 | 30.92 | SGLang `/v1/stats` unavailable; first completion warmup took 49s and was excluded. |
| Qwen3-4B | ARLE | 146.25 | 12315.8 | 71.66 | 30.73 | service trace ok; kv_util peak 97.0%. |
| Qwen3.5-4B | SGLang | 151.06 | 7559.2 | 75.77 | 27.40 | Qwen3.5 disabled overlap schedule and piecewise CUDA graph; 12 incomplete output tokens at window end. |
| Qwen3.5-4B | ARLE | 158.46 | 10897.4 | 63.22 | 24.07 | service trace ok; 3290 incomplete output tokens at window end. |

## Deltas

| Comparison | out tok/s | TTFT p50 | ITL p50 | E2E mean |
|---|---:|---:|---:|---:|
| ARLE Qwen3 vs same-run SGLang | +9.4% | +47.0% slower | -16.9% faster | -0.6% faster |
| ARLE Qwen3.5 vs same-run SGLang | +4.9% | +44.2% slower | -16.6% faster | -12.2% faster |
| ARLE Qwen3 vs `b1cbed19` Qwen3 | +1.3% | -0.1% faster | -1.0% faster | -0.9% faster |
| ARLE Qwen3.5 vs `b1cbed19` Qwen3.5 | +2.4% | -8.5% faster | +7.9% slower | -0.3% faster |

Baseline entries from `b1cbed19`:

- `docs/experience/wins/2026-04-29-bench-guidellm-cuda-l4-qwen3-c16-post-step-phase.md`
- `docs/experience/wins/2026-04-29-bench-guidellm-cuda-l4-qwen35-c16-post-step-phase.md`

GPU `nvidia-smi dmon` samples:

| Run | samples | avg SM util | avg mem util | avg power W |
|---|---:|---:|---:|---:|
| SGLang Qwen3 | 160 | 76.2% | 57.0% | 63.4 |
| ARLE Qwen3 | 159 | 75.4% | 56.8% | 62.3 |
| SGLang Qwen3.5 | 159 | 73.2% | 59.4% | 62.2 |
| ARLE Qwen3.5 | 159 | 74.8% | 55.5% | 62.4 |

## Short Request Gate

- Implemented short prompt bypass threshold `256` tokens.
- Verification burst: 16 concurrent prompts of 32 tokens, `max_tokens=16`,
  on ARLE Qwen3.5 after the main bench.
- Logs showed short requests admitted as 32-token cold prefill and the batch
  step was `plan=prefill`, not `plan=decode+prefill`.
- The measured short burst TTFT had a bad tail because GuideLLM had just left
  two long requests active and cleanup demoted prefix cache pages before most
  short requests admitted. This confirms the earlier feedback: short-prompt
  validation must run on a clean short-only workload, not immediately after a
  long-prompt bench tail.

## Problems

- Installing SGLang replaced torch/flashinfer Python packages. ARLE release
  build still succeeded after reinstall because FlashInfer headers remained at
  `/usr/local/lib/python3.12/dist-packages/flashinfer/data/include`.
- SGLang Qwen3 first completions request took 49s after server startup, even
  after graph capture and server ready. A manual warmup request was required
  before the GuideLLM window.
- SGLang does not expose ARLE's `/v1/stats`; wrapper service trace fields are
  `n/a` for SGLang.
- `nsys` is not installed on this host. CPU/GPU overlap profiling used
  scheduler step-phase logs and `nvidia-smi dmon`; proper timeline profiling
  remains pending Nsight availability.

## Learnings

- ARLE decode ITL is ahead of SGLang on both models in this setup, but TTFT is
  still materially worse. The next likely target is prefill/admission/cleanup
  scheduling, not decode kernel throughput.
- Qwen3.5 comparisons are affected by implementation asymmetry: SGLang disabled
  overlap schedule and piecewise CUDA graph on this model, while ARLE retained
  piecewise decode graph capture.
- Long-prompt benches can leave active tail requests after the GuideLLM window;
  short-prompt checks must be isolated to avoid measuring leftover long cleanup.
