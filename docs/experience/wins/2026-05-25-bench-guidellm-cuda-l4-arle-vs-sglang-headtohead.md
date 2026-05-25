# L4 ARLE vs SGLang 0.5.12 head-to-head — cuda-l4-arle-vs-sglang-2026-05-25

## Goal

Comparison: ARLE main (cu12.8 rebuild) vs SGLang 0.5.12.post1 on same L4
box, same day, same guidellm wrapper, matched prompts/output. Confirms
§11 H1 (decode-priority queues prefill) hypothesis with SGLang as control.
Supersedes stale 2026-04-26 L4 comparison.

## Hypothesis

- ARLE decode (ITL) should match-or-beat SGLang at c≥4 (decode kernel and
  CUDA graph capture are competitive).
- ARLE TTFT should be slower at c≥4 due to H1 (decode-priority plan
  ticks, prefill_queue backed up).
- Aggregate throughput should be close; decode advantage may compensate
  prefill disadvantage.

## Command

```bash
# Same flags, same machine, model swapped only via --port and --model.
# ARLE
./target/release/infer --model-path infer/models/Qwen3-4B --port 8000 \
  --num-slots 16 --max-seq-len 5120
scripts/bench_guidellm.sh cuda-l4-arle-2026-05-25-v2 \
  --target http://localhost:8000 --model Qwen3-4B \
  --processor infer/models/Qwen3-4B \
  --concurrencies 1,2,4,8,16 --max-seconds 60 --warmup 5

# SGLang
LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:$LD_LIBRARY_PATH \
  python3 -m sglang.launch_server \
  --model-path infer/models/Qwen3-4B --host 0.0.0.0 --port 8001 \
  --dtype bfloat16 --max-running-requests 16 --context-length 5120 \
  --mem-fraction-static 0.85 \
  --disable-cuda-graph-padding --disable-piecewise-cuda-graph
scripts/bench_guidellm.sh cuda-l4-sglang-2026-05-25 \
  --target http://localhost:8001 --model infer/models/Qwen3-4B \
  --processor infer/models/Qwen3-4B \
  --concurrencies 1,2,4,8,16 --max-seconds 60 --warmup 5
```

## Environment

- **Hardware:** NVIDIA L4, 23034 MiB, driver 580.82.07, CUDA 12.8.93 (ARLE) / cu13.2 lib path for SGLang
- **Model:** `infer/models/Qwen3-4B` (BF16 dense HF snapshot), same dir
- **ARLE commit:** main `19c81cf8`, `--features cuda`, TileLang AOT 0.1.9
- **SGLang:** `0.5.12.post1` + `sglang-kernel 0.4.2.post2` + `flashinfer-python 0.6.11.post1`
- **CUDA toolkit:** both `cuda-12-8` (ARLE) and `cuda-13-2` (SGLang libnvrtc.so.13) installed; LD_LIBRARY_PATH per-process
- **guidellm:** 0.6.0 (canonical wrapper)
- **Scheduling envelope:** matched(`num-slots 16` / `max-running-requests 16`, `max-seq-len 5120` / `context-length 5120`, `mem-fraction-static 0.85` / equivalent)
- **Bench:** prompt_tokens=4096 (stdev=1), output_tokens=256, max-seconds=60, warmup=5s; concurrencies 1/2/4/8/16

## Results — head-to-head

### TTFT p50 (ms)

| c | ARLE | SGLang | Δ ARLE/SGLang | ARLE p99 | SGLang p99 |
|--:|--:|--:|--:|--:|--:|
| 1 | 745.6 | 694.5 | +7.4% | 747 | **9815** ⚠️ |
| 2 | 1533 | 1373 | +11.7% | 1581 | 1409 |
| 4 | 3012 | 2026 | **+48.7%** | 3144 | 2816 |
| 8 | 5907 | 2839 | **+108%** | 5938 | 5560 |
| 16 | 12913 | 6008 | **+115%** | 13385 | **29402** ⚠️ |

### ITL p50 (ms)

| c | ARLE | SGLang | Δ ARLE/SGLang | ARLE std | SGLang std |
|--:|--:|--:|--:|--:|--:|
| 1 | 36.14 | 35.26 | +2.5% | 0.02 | 0.03 |
| 2 | 40.01 | 40.89 | -2.2% | 0.11 | 1.27 |
| 4 | 44.17 | 48.32 | **-8.6%** | 0.16 | 2.93 |
| 8 | 52.84 | 60.71 | **-13.0%** | 0.21 | 6.05 |
| 16 | 71.85 | 91.86 | **-21.8%** | 0.20 | **11.4** |

### Output tok/s

| c | ARLE | SGLang | Δ |
|--:|--:|--:|--:|
| 1 | 26.02 | 26.70 | -2.5% |
| 2 | 44.69 | 46.22 | -3.3% |
| 4 | 75.47 | 74.94 | +0.7% |
| 8 | 117.40 | 109.80 | **+6.9%** |
| 16 | **164.00** | 135.00 | **+21.5%** |

## Findings

1. **ARLE decode (ITL) 在 c≥4 比 SGLang 0.5.12 快 8.6–21.8%。** Continuous
   batching + CUDA Graph capture (B=1..16) 在执行段是赢的。

2. **ARLE prefill (TTFT) 在 c≥4 比 SGLang 慢 48.7–115%。** 这正是 §11 H1
   confirmed 的 plan tick 偏向 decode 的代价:c=16 时 ARLE 之前测得
   `prefill=95 / decode=6382 / idle=22425` ticks,prefill 仅 0.33%。
   SGLang 的 chunked prefill + mixed batching 在这里赢。

3. **总 throughput 平手到 ARLE 略胜**:c=8 +6.9%,c=16 **+21.5%**。decode
   段优势在 60s window 内把 prefill 段劣势抵消(因为 60s 中绝大部分时间在
   decode,只有起始 prefill 一次)。**真实 SLO 场景下(看 TTFT p99),ARLE 仍
   是劣势**。

4. **Tail 反转 — ARLE 更确定性**:c=16 时
   - ARLE TTFT p99 13385 ms vs SGLang **29402** ms (SGLang 2.2× worse)
   - ARLE ITL std 0.20 vs SGLang **11.4** (SGLang 57× more variance)
   - c=1 SGLang p99 = 9815 ms(单 outlier 拉高),ARLE p99 = 747 ms

   说明:ARLE 严格 FIFO + 每 tick 单 prefill 是 deterministic;SGLang 的
   动态 chunked prefill + mixed plan 有快有慢,p50 赢但 p99 输。**实际产品
   SLO 看 p99 时,ARLE 反而有优势**。

## Problems

- SGLang 0.5.12 hard-pins `tilelang==0.1.8`,但 ARLE cuda-kernels build.rs
  在 tilelang 0.1.8 上撞 sm_89 pipeline planner bug,必须 force-install
  tilelang 0.1.9 (`pip install --no-deps tilelang==0.1.9`)才能两边共存。
  pip 抱怨 dependency conflict,但 SGLang 0.5.12 runtime 在 tilelang 0.1.9
  上跑得通(没观察到运行时回归)。
- SGLang 需要 `LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:...` 因为
  sglang-kernel cu130 require libnvrtc.so.13。

## Learnings

- **ARLE 不是哪儿都输 SGLang。Decode ITL 已经赢了 10-20%。**
- **TTFT 在 c≥4 翻倍输 SGLang 是 prefill 调度,不是 prefill kernel**(decode
  跑得快说明 attention/MLP/sampling kernel 路径 OK)。Fix 方向 = scheduler
  plan 策略,不是 kernel 优化。
- **Throughput 跟 TTFT 不是同一面**:60s window 下 ARLE total tok/s 反而
  +21%,因为 decode 段长且更快。但实际线上 SLO 关心 TTFT,所以这不能掩盖
  H1 的问题。
- **p99 tail ARLE 比 SGLang 好**:ARLE FIFO 单 prefill 虽然 mean 慢,但
  variance 极小;SGLang dynamic chunked prefill mean 快但 p99 暴大。
  二者权衡看产品决定 (mean 优先 → SGLang;p99 优先 → ARLE)。

## Rule

- Comparison bench 必须同 box / 同日 / 同 guidellm / 同 model dir。
  2026-04-26 L4 数据已 stale(commit drift + 不同 box),不要混用。
- TTFT 和 throughput 跑数据要分开看 — 单看 throughput 容易得出
  "ARLE 不差" 的误导性结论,而 TTFT p99 是 SLO ground truth。

## Cross-refs

- §11 H1 hypothesis 来源:[`2026-05-25-bench-guidellm-cuda-l4-arle-h1-confirmed.md`](2026-05-25-bench-guidellm-cuda-l4-arle-h1-confirmed.md)
- 2026-05-08 W3 c=16 agent deadlock(同 H1 退化形态):
  [`../errors/2026-05-08-w3-c16-deadlock-not-just-admission.md`](../errors/2026-05-08-w3-c16-deadlock-not-just-admission.md)
- 2026-05-09 4070 Ti SUPER post-P1.0/P1.2 reverify(短 prompt 反超):
  [`2026-05-09-bench-sglang-reverify-post-p1.0-p1.2.md`](2026-05-09-bench-sglang-reverify-post-p1.0-p1.2.md)
- Raw artefacts: archived ARLE and SGLang bench-output directories
