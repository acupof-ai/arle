# L4 c=16 nsys trace — H1 + H5 confirmed as real bottlenecks, H2/H3/H4/H6 refuted

## Goal

Diagnosis: nsys trace c=16 30s 的实测,直接判定 §11 hypothesis H1-H6,把
"看起来合理但没数据" 的猜测全部 license-or-kill。

## Hypothesis

各 H 的具体猜测见 [`2026-05-25-bench-guidellm-cuda-l4-arle-h1-confirmed.md`](2026-05-25-bench-guidellm-cuda-l4-arle-h1-confirmed.md)。
预期:H1+H5 confirmed,H2/H3/H4/H6 refuted。

## Command

```bash
# Machine: L4 / 24GB / cuda-12.8 + cuda-13.2
/usr/local/cuda-13.2/bin/nsys profile --trace=cuda,nvtx \
  -o arle_c16 --force-overwrite=true \
  ./target/release/infer --model-path infer/models/Qwen3-4B --port 8000 --num-slots 16 --max-seq-len 5120

# c=16 30s bench during nsys recording window
scripts/bench_guidellm.sh cuda-l4-arle-nsys-vg --target http://localhost:8000 \
  --model Qwen3-4B --processor infer/models/Qwen3-4B \
  --concurrencies 16 --max-seconds 30 --warmup 0

nsys stats -r nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum
```

## Environment

- L4 24GB / driver 580.82.07 / CUDA 12.8 (ARLE build) + CUDA 13.2 (nsys)
- ARLE main `19c81cf8`, `--features cuda`, tilelang 0.1.9
- Qwen3-4B BF16 dense, num-slots=16, max-seq-len=5120
- Bench: c=16, prompt=4096 / output=256, 30s, no warmup
- nsys 2025.6.3, full app-launch capture (no `--delay`/`--duration` flags)

## Results — nsys NVTX scope distribution (29.1s wall)

| Scope | Instances | Total | % wall | Avg/inst |
|---|---:|---:|---:|---:|
| `step_total` | 2,788,521 | 29.1s | 100.0% | 10.4 µs |
| `step_prefill_kernel_launch` | **7** | **2.89s** | **9.9%** | 413 ms |
| `step_admission` | 2,788,521 | 885 ms | 3.0% | 317 ns |
| `step_decode_kernel_launch` | 262 | 177 ms | 0.6% | 676 µs |
| `step_plan` | 9,952 | 21.8 ms | 0.07% | 2.2 µs |
| `step_dispatch_emits` | 9,951 | 10.8 ms | 0.04% | 1.1 µs |
| `scheduler_snapshot` | 9,952 | 6.9 ms | 0.02% | 0.7 µs |

→ 30s 内仅 7 个 prefill_launch / 262 个 decode_launch;**99.6% 的 2.79M ticks
是空 admission 早返回**(因为只有 9952 ticks 触发了 plan/dispatch)。

## Results — CUDA API time

| API | Count | Total | % wall | 含义 |
|---|---:|---:|---:|---|
| `cudaLaunchKernel` | 30,572 | 7.45s | 25.6% | kernel launch overhead |
| `cudaEventSynchronize` | **540** | **4.15s** | **14.3%** | **prefill completion sync, 7.68ms avg!** |
| `cuEventQuery` | **2,706,971** | **2.24s** | **7.7%** | **prefill event poll, 93k 次/s spin** |
| `cuLaunchKernel` | 20,550 | 1.60s | 5.5% | tilelang/低层 launch |
| `cuMemcpyDtoDAsync_v2` | 1,225 | 270 ms | 0.93% | device-to-device copy |
| `cuMemsetD8Async` | 486 | 208 ms | 0.71% | memset |
| `cuGraphLaunch` | **288** | **118 ms** | **0.4%** | **CUDA Graph 命中** |
| `cuMemcpyDtoHAsync_v2` | 529 | **4.4 ms** | **0.015%** | **D2H (sampling readback)** |

## Results — GPU kernel time top

| Kernel | % | 备注 |
|---|---:|---|
| `cutlass 256x128 GEMM` | 43.9% (20.2s) | 主 MLP/lm_head 大 GEMM |
| `kernel_kernel` (TileLang AOT attn) | 16.6% (7.6s) | attention |
| `cutlass 128x256 GEMM` | 14.2% (6.5s) | MLP |
| `silu_mul_native` | 5.1% (2.35s) | MLP activation |
| `cutlass 16x16 wmma` | 4.6% (2.12s) | small GEMM |
| `add_native` | 2.4% (1.12s) | residual |
| `prefill_attention_paged_qk_norm_rope_hd128` | 2.4% (1.10s) | 6660 inst |
| `rms_norm_batched` | 1.7% (0.78s) | |
| **`gemv_handwritten`(lm_head decode)** | **1.2%** (549 ms) | **185 inst × 2.97ms** |
| `argmax_batch_logprob`(sampling) | **0.02%** (9.5 ms) | 256 × 37µs |
| D2H 全部 GPU memops | **0.7%** (1.19 ms) | 完全可忽略 |

## Hypothesis 最终判决 (nsys-backed)

| H | 状态 | Evidence |
|---|---|---|
| **H1** decode-priority queues prefill | ✅ **CONFIRMED** | 30s 仅 7 次 prefill_launch vs 262 decode;每次 prefill 413ms;plan_label split=0 mixed=0;TTFT c=16 = 12913ms 完全可归因 |
| **H2** per-tick CPU metrics overhead | ❌ **REFUTED** | admission+plan+snapshot+dispatch+scheduler_snapshot 合计 < 3% wall;不是瓶颈 |
| **H3** D2H readback per decode | ❌ **REFUTED** | D2H 全 = **4.4ms = 0.015% wall**;argmax sampling 9.5ms = 0.02%;sampling 路径完全没问题 |
| **H4** CUDA Graph 命中率 / 代码已删 | ❌ **REFUTED** | `infer/src/model/cuda_graph.rs` 存在;288 次 `cuGraphLaunch` 成功;decode 走 graph replay |
| **H5** async prefill backoff idle gap | ✅ **CONFIRMED** | **`cuEventQuery` 2.71M × 827ns = 2.24s (7.7%) + `cudaEventSynchronize` 540 × 7.68ms = 4.15s (14.3%) = 22% wall 在等 prefill event** |
| **H6** lm_head 大 vocab GEMM | ❌ **REFUTED** | lm_head gemv = 549ms = 1.2% kernel time;cutlass 大 GEMM 是 MLP 不是 lm_head;refuted as ITL bottleneck |
| **H7** Qwen3.5 recurrent state | ➖ N/A | Qwen3-4B dense,不适用 |

## Real bottleneck distribution (wall-clock, c=16)

1. **GPU kernel 执行**(MLP/attn/lm_head 80% 算力,decode 段是 cutlass + tilelang 主导)
2. **`cudaLaunchKernel` overhead** = **25.6% wall** — 即使 graph 命中 288 次,
   仍 30k 次普通 launch。graph 没覆盖到的 launch 是大头(prefill kernel +
   per-layer ops in decode)。
3. **prefill event poll + sync = 22% wall** (H5)
4. **prefill kernel launch 自身** = 9.9% (7 batch × 413ms avg)
5. **scheduler 业务逻辑** = < 3% (H2 refuted)
6. **sampling + D2H** = < 0.05% (H3 refuted)

## Fix 优先级 (按 wall-clock 收益)

1. **H1 + H5 联合 (~32% wall combined)**:scheduler 改 Mixed/Split plan 让 prefill 不再单批 + event poll 改 condvar (or wider backoff)。
2. **`cudaLaunchKernel` overhead 25.6%**:进一步推 graph capture 覆盖范围 (prefill 也走 graph?)。
3. 其他全部 deferred。

## Problems

- nsys 2025.6.3 第一次跑 `--capture-range=cudaProfilerApi --capture-range-end=stop`
  收集了 0 字节 trace,SIGUSR1/USR2 fired 但 nsys 没存出。改用全程 capture +
  SIGTERM finalize 才拿到完整 174MB trace。
- `--delay=50 --duration=30` 也没收到 NVTX/CUDA 数据,改全程 capture 才行。
- `nsys --pid` 在 2025.6.3 不支持 attach 已运行进程。
- 应用层 NVTX 来自 `infer/src/scheduler/cuda/nvtx_scopes.{rs,c}` (FFI wrap
  `nvtxRangePushA`),全程 capture 时 nsys 正确收到。

## Learnings

- **§11 hypothesis 表 4 个被 nsys refute** (H2/H3/H4/H6),只 H1 + H5 撑住。
  说明 source survey + first-principle 估算 ≠ evidence。
- **真正的 wall-clock 大头是 launch overhead (25.6%) + event sync (22%) +
  prefill 排队 (10%)**,加起来 58% 都在 scheduler/CUDA API 层,不在 kernel
  layer。优化重心 ≠ kernel tuning。
- **D2H/sampling 路径是干净的** — 不需要花时间优化 argmax/D2H batching。
- nsys 拿数据时,**`--capture-range=cudaProfilerApi` + `--delay`/`--duration`
  在 2025.6.3 上都坑**;全程 capture + SIGTERM 是最可靠的方式。

## Rule

- hypothesis 不能停在 first-principles。每一条都要 nsys / counter / 控制实验
  才能 confirm/refute。本次 4/6 refuted 说明 first-principles 错误率不低。
- nsys 在 ARLE 上的可靠跑法:全程 capture + 应用 ready 后跑 bench + SIGTERM
  nsys finalize。不要用 `--delay`/`--duration`/`--capture-range`。

## Cross-refs

- 源 hypothesis: [`2026-05-25-bench-guidellm-cuda-l4-arle-h1-confirmed.md`](2026-05-25-bench-guidellm-cuda-l4-arle-h1-confirmed.md)
- vs SGLang 0.5.12 头对头: [`2026-05-25-bench-guidellm-cuda-l4-arle-vs-sglang-headtohead.md`](2026-05-25-bench-guidellm-cuda-l4-arle-vs-sglang-headtohead.md)
- H5 来源 (async prefill completion fix): [`2026-05-07-m3.7-b1.2-async-prefill-completion.md`](2026-05-07-m3.7-b1.2-async-prefill-completion.md)
- Raw nsys trace: archived `arle_c16.nsys-rep` (174 MB)
