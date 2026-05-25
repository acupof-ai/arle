# L4 c-sweep + §11 hypothesis verification — cuda-l4-arle-2026-05-25

## Goal

Diagnosis: validate the §11 high-concurrency hypotheses (see Lark
"ARLE 架构梳理" doc) against real ARLE bench data on an L4 machine.
Refute or confirm each H1–H7 with file:line / counter evidence instead of
inference. SOLID 第一原则:source survey 是 hypothesis,evidence = bench
counter / log。

## Hypothesis

- **H1**(decode-priority queues prefill)应该是最强嫌疑;预期 plan label
  ticks 上看 prefill 远低于 decode + prefill_queue 堆积。
- **H4**(CUDA graph 可能已删)应该 refuted,因为 grep `cuda_graph.rs`
  和 `cudarc::CudaGraph` 都有命中。
- **H6**(lm_head 大 vocab GEMM 是 ITL 主因)应该 refuted,first-principle
  算 ~73us / step 远小于 ~71ms 的 step time。
- H2 / H3 / H5 需更细 nsys 才能定量,这次只做信号检测。
- H7 N/A(Qwen3-4B dense,不走 Qwen3.5 recurrent state)。

## Command

```bash
# Machine: L4, 24GB, CUDA 12.8.93
cd agent-infer
export CUDA_HOME=/usr/local/cuda PATH=$CUDA_HOME/bin:$PATH \
  LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH TORCH_CUDA_ARCH_LIST=8.9
./target/release/infer --model-path infer/models/Qwen3-4B \
  --port 8000 --num-slots 16 --max-seq-len 5120

scripts/bench_guidellm.sh cuda-l4-arle-2026-05-25 \
  --target http://localhost:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B \
  --concurrencies 1,2,4,8,16 \
  --max-seconds 60 --warmup 5
```

## Environment

- **Backend:** CUDA, `--features cuda` (no tilelang-attn opt-in feature)
- **Model:** `infer/models/Qwen3-4B` (BF16 dense; HF snapshot)
- **Hardware:** NVIDIA L4, 23034 MiB, driver 580.82.07, CUDA 12.8.93
- **Commit:** main `19c81cf8`
- **Feature set:** `cargo build --release -p infer --features cuda`
- **Non-default flags:** `--num-slots 16`, `--max-seq-len 5120`
- **TileLang:** pinned to 0.1.9 (0.1.8 fails sm_89 pipeline planner)
- **Scheduling envelope:** `max_num_batched_tokens=16384`,
  `chunked_prefill_size=2048`, `max_prefill_tokens=16384`,
  `mem_fraction_static=0.85`, `max_slots=16`

## Results — c-sweep headline

| c | TTFT p50 (ms) | TTFT p99 | ITL p50 | ITL p99 | out tok/s | req/s | TTFT vs c=1 | ITL vs c=1 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 722.5 | 732.9 | 35.91 | 35.97 | 26.18 | 0.109 | 1.00× | 1.00× |
| 2 | 1462.4 | 1525.3 | 39.60 | 39.78 | 45.25 | 0.182 | 2.02× | 1.10× |
| 4 | 2930.6 | 2950.7 | 43.72 | 44.01 | 75.82 | 0.291 | 4.05× | 1.22× |
| 8 | 5734.4 | 5764.7 | 51.94 | 52.46 | 119.67 | 0.436 | 7.94× | 1.45× |
| **16** | **12712.1** | **12825.3** | **71.36** | **71.57** | **166.62** | 0.291 | **17.6×** | **1.99×** |

**TTFT 接近线性放大(17.6× @ c=16),ITL 仅 1.99× → 瓶颈在 prefill 调度
而非 decode kernel**(与 2026-04-26 L4 bench 同一形态)。

## Service trace peaks (c-sweep 总计 60s × 5 个 rate, 453 个 1s sample)

| 指标 | 值 |
|---|---|
| Peak active | 16 |
| Peak running_batch | 16 |
| **Peak prefill_queue** | **15** (16 sessions, 15 排队等 prefill) |
| Peak kv_util | 100.0% |
| **Plan tick 分布** | **idle=22425 / decode=6382 / prefill=95 / split=0 / mixed=0** |
| Prefill plan tick 占比 | **0.33%** (95 / 28902) |
| Decode plan tick 占比 | 22.1% |
| Idle plan tick 占比 | 77.6% |
| KV fetch waiter samples >0 | 0/453 (KV tier 无压力) |
| step_phase_us (after) | adm:340 / prefill:0 / decode:5 / total:345 / cleanup:22 |
| readback_not_ready 累计 | 67,662,957 (高频但单次极短) |

## Hypothesis verdict

| H | 假设 | 状态 | 数据依据 |
|--|---|---|---|
| **H1** | Decode-priority queues prefill | ✅ **CONFIRMED (smoking gun)** | Plan tick 分布:**prefill 仅 0.33% / decode 22% / idle 78%**;split=mixed=0 表示 scheduler 完全没把 prefill 插进 decode batch;peak prefill_queue=15;TTFT c=1→16 17.6× / ITL 1.99×。决定性。 |
| **H2** | per-tick CPU metrics overhead | ⚠️ **PARTIAL** | step_phase_us total=345us, loop_total=336us(单 tick 实际工作 ~0.34ms);idle tick 374/s × ~0.3ms ≈ **11% CPU on 空 tick**;真实但非主因。隔离需 env flag ablation(待补)。 |
| **H3** | D2H readback per decode step | ❓ **NEEDS NSYS** | gpu_completion_wait=0 in trace,decode kernel=5us / tick;ITL 71ms 余下 ~66ms 不能仅凭 trace 归因到 D2H。需 nsys NVTX 分段。 |
| **H4** | CUDA graph 命中率 / 代码已删 | ❌ **REFUTED** | `infer/src/model/cuda_graph.rs` 存在(`CudaGraphState` wrap `cudarc::driver::safe::CudaGraph`);log 中 warmup 期间 **`Capturing CUDA Graph for batched decode B=1..16` 全部成功**;startup `cuda_graph=true`;Qwen3 `supports_cuda_graph_decode()=true`。launch overhead 不是问题。 |
| **H5** | prefill backoff idle gap | ❓ **NEEDS NSYS** | trace 看到 prefill_queue=15(已被 H1 解释),无法分离 backoff idle vs decode-priority 排队。 |
| **H6** | lm_head 大 vocab GEMM 占 ITL 大头 | ❌ **REFUTED (first-principles)** | Qwen3-4B vocab 152K × hidden 3584 × B=16 = 8.7 GFLOPs → L4 @ ~120 TFLOPS BF16 理论 73us;实际 step ~71ms(B=16);lm_head < 1% of step。decode 是 memory-bound(32 层 × attn+mlp 主导),不是 compute-bound lm_head。 |
| **H7** | Qwen3.5 recurrent state copy | ➖ **N/A** | 跑的是 Qwen3-4B dense,recurrent state 路径不触发。 |

## Problems / 异常

- **TileLang 0.1.8 默认 pin 在 sm_89 上 build 失败**(`pipeline planner: 14 stages != 15 pipeline stages` TVM internal error)。手动 `pip install tilelang==0.1.9` 才通。`pyproject.toml` 的 `tilelang>=0.1` 太宽。建议:pin `tilelang==0.1.9` 或 vendored AOT cache(讨论中)。
- CUDA build 首次耗时 14m15s(主要是 cuda-kernels TileLang Python codegen + 32 个 .o nvcc 编译);考虑 vendor TileLang `.c` 进 git 跳过 Python 步骤。

## Learnings

- **H1 是首要敌人,不是 launch / kernel / D2H**:scheduler 的 plan 策略
  在 c=16 时 95% 偏向 decode,prefill 100% 走串行单 tick(split=0,mixed=0)
  → 新请求 TTFT 线性堆积。Fix 方向是 admission policy + 主动 Mixed/Split
  plan 触发,不是去优化 prefill kernel。
- **CUDA graph 已生效**(B=1..16 全部 capture),进一步 launch overhead
  优化收益有限。
- **lm_head 不是大 ITL 项**:即使 vocab 大,Qwen3-4B decode 是 32 层
  attn+mlp memory-bound 主导;ITL 减半的关键不在 sampling/lm_head。
- **H2 信号弱但真实**:11% CPU 跑空 idle tick,长尾可优化但非主战场。
- **TTFT 线性放大 ≈ prefill 串行**:c=N 时 TTFT 约 N×base,正是 H1 描述的
  "新请求轮流单独 prefill"。

## Rule

- 假设要 license-or-kill:H4/H6 用 first-principles / grep 直接 refute,
  H1 用 plan tick counter 直接 confirm。**不要因为"看起来合理"就保留 hypothesis**。
- 在 wins 写 hypothesis 验证时,verdict 必须配 counter 数字或 git 路径,
  不能只写"refuted",必须说明是哪条 evidence 让它 refuted。

## Cross-refs

- 飞书 §11 hypothesis 来源 + 上一次 stale baseline:
  [`2026-04-26-bench-guidellm-cuda-l4-vs-sglang-c1-c16.md`](2026-04-26-bench-guidellm-cuda-l4-vs-sglang-c1-c16.md)
- 2026-05-08 W3 c=16 deadlock(同一 H1 现象的退化形态):
  [`errors/2026-05-08-w3-c16-deadlock-not-just-admission.md`](../errors/2026-05-08-w3-c16-deadlock-not-just-admission.md)
- **后续 follow-up(同日)**:
  - vs SGLang 0.5.12 头对头: [`2026-05-25-bench-guidellm-cuda-l4-arle-vs-sglang-headtohead.md`](2026-05-25-bench-guidellm-cuda-l4-arle-vs-sglang-headtohead.md) — 同 box 对照,decode ITL 反超,prefill TTFT 仍 +115%
  - nsys 实测 H1-H7 verdict: [`2026-05-25-bench-guidellm-cuda-l4-arle-nsys-h1-h5-confirmed.md`](2026-05-25-bench-guidellm-cuda-l4-arle-nsys-h1-h5-confirmed.md) — H1+H5 CONFIRMED,H2/H3/H4/H6 REFUTED,真正 wall 大头是 prefill 排队 + event poll + launch overhead
  - INFER_PREFILL_GRAPH=1 默认化 KILL: [`errors/2026-05-25-prefill-graph-default-kill.md`](../errors/2026-05-25-prefill-graph-default-kill.md) — graph cache 8-slot 在 c=16 thrash → -86% tok/s
- Raw artefacts: archived bench-output directory
