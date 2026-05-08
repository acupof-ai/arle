# 2026-05-08 · M_pf-graph v2 Phase 0v2.A — nsys Baseline Gate

## Goal

Diagnosis gate for [`M_pf-graph v2`](../../plans/M_pf-graph-v2.md): measure whether production auto-FP8 4k/c=4 prefill is launch-dispatch bound enough to license Phase 0v2.B.

## Hypothesis

If R1's SGLang finding in the master strategy is correct, ARLE's 4k prefill should show high launch density and CUDA launch host time should be at least 30% of the prefill launch step window. If dispatch is below 30%, Phase 0v2 should be killed before re-implementing the graph substrate.

## Command

Wrapper migration note: `scripts/profile_nsys_guidellm.sh` still documents the old attach flow and rejects nsys 2024+ because `--attach-pid` is gone. The working path was `scripts/profile_nsys_signal.sh`, which spawns ARLE under `nsys --capture-range=cudaProfilerApi` and toggles capture with SIGUSR1/SIGUSR2. This run added `--data` forwarding to that wrapper so the profile uses the exact 4096/256 long-context shape.

```bash
CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/profile_nsys_signal.sh m_pf_graph_v2_baseline \
  --server-args "--model-path infer/models/Qwen3-4B --port 8000 --num-slots 8 --max-seq-len 5120" \
  --concurrencies 4 --max-seconds 60 \
  --data 'prompt_tokens=4096,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_min=256,output_tokens_max=256'
```

Generated nsys command:

```bash
nsys profile \
  --output bench-output/2026-05-08-m_pf_graph_v2_baseline-profile-nsys-signal/trace \
  --force-overwrite=true \
  --trace cuda,nvtx,osrt \
  --cuda-graph-trace node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --export=sqlite \
  --kill none \
  -- target/release/infer \
    --model-path infer/models/Qwen3-4B \
    --port 8000 --num-slots 8 --max-seq-len 5120
```

## Environment

| Field | Value |
|---|---|
| GPU | RTX 4070 Ti SUPER 16 GiB (`sm_89`) |
| CUDA | `/opt/cuda`, Nsight Systems `2025.6.3.541-256337736014v0` |
| Runtime commit at trace start | `d54a346` |
| Current doc commit context | `3fe25e0` (later docs-only strategy updates) |
| Model | `infer/models/Qwen3-4B` |
| KV mode | production `auto` → paged pool `FP8E4M3` |
| Feature set | release CUDA binary, `cuda_graph=true` |

Scheduling envelope:

```text
Scheduling envelope (resolved | SGLang-equiv): max_num_batched_tokens=16384 | 16384, chunked_prefill_size=2048 | 2048, max_prefill_tokens=16384 | 16384, mem_fraction_static=0.85 | 0.85, max_slots=8 | (n/a — SGLang has no fixed cap)
Config: model_path=infer/models/Qwen3-4B, cuda_graph=true, num_slots=8 (explicit), kv_cache_mode=auto (auto-fp8)
KV cache layout: contiguous=BF16, paged_pool=FP8E4M3
```

## Results

Raw artifacts:

- `bench-output/2026-05-08-m_pf_graph_v2_baseline/`
- `bench-output/2026-05-08-m_pf_graph_v2_baseline-profile-nsys-signal/trace.nsys-rep`
- `bench-output/2026-05-08-m_pf_graph_v2_baseline-profile-nsys-signal/trace.sqlite`
- `bench-output/2026-05-08-m_pf_graph_v2_baseline-profile-nsys-signal/cuda_api_sum.txt`
- `bench-output/2026-05-08-m_pf_graph_v2_baseline-profile-nsys-signal/cuda_gpu_kern_sum.txt`

GuideLLM anchor:

| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E mean | conc p50 | out tok/s | total out | req/s actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conc4 | 2033.1 ms | 88.8 ms | 1995.9 ms | 2287.6 ms | 19.36 ms | 19.50 ms | 6.97 s | 4 | 152.32 | 9216 | 0.533 |

Service trace:

```text
Peak active: 4
Peak running_batch: 4
Peak prefill_queue: 3
Plan labels: idle=15508, decode=2302, prefill=31, split=0, mixed=0
Peak kv_util: 85.9%
Prefix hit rate: peak 0.0%, q75 0.0%
```

### nsys CUDA API Summary

Whole capture headline:

| API | Calls | Host total | Avg |
|---|---:|---:|---:|
| `cuEventQuery` | 13,462,715 | 8.352 s | 0.620 us |
| `cuGraphLaunch` | 2,302 | 2.587 s | 1.124 ms |
| `cudaLaunchKernel` | 20,084 | 197.852 ms | 9.851 us |
| `cuLaunchKernel` | 8,604 | 51.633 ms | 6.001 us |
| `cudaLaunchKernelExC` | 1,440 | 8.705 ms | 6.045 us |

Prefill-only `step_prefill_kernel_launch` NVTX window:

| Metric | Value |
|---|---:|
| NVTX prefill launch ranges | 31 |
| Total prefill launch range wall time | 342.565 ms |
| Launch API calls inside those ranges | 27,789 |
| Launch API host time inside those ranges | 190.778 ms |
| Launch host time / prefill launch range | **55.7%** |
| Conservative launch density over 60s bench | **463 launches/sec** |
| Density inside prefill launch ranges | 81.1k launches/sec |

License gate:

| Gate | Threshold | Measured | Decision |
|---|---:|---:|---|
| Launch density | ≥ 200 launches/sec | 463/sec over bench | PASS |
| Launch host time / prefill step window | ≥ 30% | 55.7% | PASS |
| Top launch groups identified | required | yes | PASS |

### Top Launch Groups By Host Time

Grouped by correlated kernel short name inside `step_prefill_kernel_launch`:

| Kernel group | Launch calls | Host launch time | GPU time fully inside range |
|---|---:|---:|---:|
| `Kernel2` (cuBLASLt/CUTLASS GEMMs) | 7,488 | 46.104 ms | 116.567 ms |
| `gemv_handwritten_kernel` | 109 | 32.711 ms | 0.000 ms |
| `prefill_attention_paged_qk_norm_rope_hd128_kernel` | 3,924 | 21.917 ms | 7.793 ms |
| `prefill_attention_paged_kv_write_hd128_kernel` | 3,924 | 21.356 ms | 2.132 ms |
| `rms_norm_batched_kernel` | 2,232 | 12.217 ms | 3.238 ms |
| `add_native_kernel` | 2,232 | 12.065 ms | 0.952 ms |
| `quantize_paged_kv_fp8_kernel` | 2,232 | 11.951 ms | 1.335 ms |

GPU kernel summary for the full capture is still dominated by GEMM/decode graph work, but Phase 0v2.A's gate is specifically the host dispatch share inside the prefill launch window.

## Problems

- The legacy attach wrapper is unusable on Nsight Systems 2025.6 because `--attach-pid` is removed. The signal wrapper is the correct path for new traces.
- `scripts/profile_nsys_signal.sh` did not forward `--data`; without the fix this run would have profiled the wrong prompt shape.
- The launch-dispatch gate passes, but it should not be interpreted as a full TTFT gap explanation. The measured removable host-launch budget is large within the prefill launch window, but only about 191 ms across the full 60s trace. Phase 0v2.B should stay license-or-kill and must prove TTFT movement on production FP8 A/B.

## Learnings

- `step_prefill_kernel_launch` is launch-dense enough to license graph work: most of the CPU-side prefill launch section is CUDA launch overhead.
- Production auto-FP8 matters. This run avoided Phase 0's BF16-forced KV distortion and kept `max_prefill_tokens=16384`, matching the SGLang-equivalent envelope.
- The next graph substrate must remove dispatch without changing scheduler admission. Phase 0's envelope clamp would have hidden or reversed this signal.

## Δ vs Baseline

Compared with the Phase 0 KILL entry [`2026-05-08-m_pgc-phase0-killed-ttft-under-threshold.md`](../errors/2026-05-08-m_pgc-phase0-killed-ttft-under-threshold.md):

| Item | Phase 0 KILL | Phase 0v2.A baseline |
|---|---|---|
| KV mode | forced BF16 | production auto-FP8 |
| Envelope | clamped to one 2048-token request | `max_prefill_tokens=16384`, no clamp |
| TTFT p50 | 1961.2 ms | 1995.9 ms |
| ITL p50 | 25.58 ms | 19.36 ms |
| out tok/s | 122.95 | 152.32 |
| nsys prefill dispatch evidence | absent | 55.7% of prefill launch range |

Decision: **license Phase 0v2.B** under the plan's prerequisite gate, with the caution that the benchmark gate still owns final promotion.
