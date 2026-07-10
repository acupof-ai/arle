# W4A8 PR #31 上游 QQQ 仓库调研 — bf16 + i4fp8 branch 升级路径

> 接续 `2026-05-09-w4a8-industry-kernel-survey.md`(确认 vLLM dequant.h 是
> W4A16 路径,不能直接用于我们 W4A8)。本 brief 调研 PR #31 直接上游
> QQQ(HandH1998)仓库,看主分支演化 + i4fp8 branch FP8 activation 支持。

## QQQ 仓库状态(2026-05-09 fetch)

仓库:`https://github.com/HandH1998/QQQ`

### 三个分支

| Branch | HEAD | 关键变化 | 我们当前 |
|--------|------|---------|---------|
| `main` | `b6582d1` add bf16 activation support | **1008 LOC**(+21 LOC vs 我们)| 987 LOC PR #31 cherry-pick(无 bf16 activation) |
| `i4fp8` | `b6582d1`(同 main commit hash 但 different code)| **1154 LOC**(+167 LOC),add FP8 activation | 不支持 |
| `bak` | (legacy) | -- | -- |

### main branch 演化(vs 我们 PR #31)

- **+21 LOC**:`add bf16 activation support`(`#include <cuda_bf16.h>`)
- 现在我们的输入 bf16 → quant int8 → mma → fp16 输出 → bf16 转换;**main 升级支持 bf16 activation 直通**(可能省 fp16↔bf16 转换 launches?)
- 需 diff 详查
- **预估 ROI**:省 1 个 launch(fp16→bf16),ITL **-2-5%**

### i4fp8 branch 演化(主要 W4 + FP8 activation 路径)

- **+167 LOC vs main**:add FP8 activation support
- sm_89 native FP8 mma(`mma.sync.aligned.m16n8k32.f32.e4m3.e4m3` peak 706 TFLOPS)
- 但需要 FP8 activation 量化路径(quantize_bf16_to_fp8 替代 quantize_bf16_to_int8)
- **优势**:FP8 numerics 比 INT8 quant 通常更稳(动态 range 大)
- **预估 ROI**:不一定显著(decode 是 memory-bound,FP8 vs INT8 mma peak 同 706 TFLOPS,无 compute 提升)
- 主要价值:**accuracy** 改善(FP8 activation 比 INT8 round 更精确)

## 三种升级选项对比

| 选项 | 来源 | LOC delta | 风险 | 预估 perf gain | 备注 |
|------|------|----------:|------|---------------|------|
| **A** | QQQ main(bf16 activation)| +21 LOC | 低 | ITL **-2-5%** | 直接抄,省转换 launch |
| **B** | QQQ i4fp8(FP8 activation)| +167 LOC + adapter | 中 | perf 0~+5%,**accuracy ↑** | 需 FP8 quant 路径 + benchmark |
| **C** | sm_89 specific tile re-tune(skill #4)| ~50 LOC + ncu sweep | 低 | ITL -5-15% | 需 ncu wrapper migration 完成 |
| D | vLLM marlin port(W4A16) | ~700 LOC | 中 | W4A16 **B5** -3-8% | 不影响 W4A8 B7,**不同 axis** |

## 推荐 Phase 顺序

### 🥇 Phase 1 — QQQ main bf16 activation(选项 A)

最低风险,纯 ~21 LOC delta,可直接 git diff 看清楚。
- **Effort**:Claude 半天 — fetch + diff + adapter 调整 + bench
- **Risk**:低(只增 bf16 path,不改既有 fp16 path)
- **预估**:B7 ITL -2-5%

### 🥈 Phase 2(可选)— sm_89 tile re-tune(选项 C)

平行轴,看完 Phase 1 数据后决定值不值。需 ncu wrapper migration 解锁。

### 🥉 Phase 3(可选)— QQQ i4fp8 FP8 activation(选项 B)

非紧急,主要 accuracy 价值。perf 不一定显著(memory-bound)。

### 🚫 不在本 axis

- vLLM marlin(选项 D)= W4A16 升级,不影响 W4A8 B7。可独立做 B5 优化。
- Machete = Hopper-only,跳过

## 立即 next step

**抄 QQQ main 的 21 LOC bf16 activation diff** → 无 model checkpoint 改变,直接 inplace 升级 W4A8 path 支持 bf16 activation,可能省 fp16↔bf16 转换 launches。

Claude 单文件 + cuda kernel diff,~1 hour wall-clock。

## Cross-references

- 上游 QQQ:https://github.com/HandH1998/QQQ
- 我们当前:`crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu`(987 LOC,基于 PR #31)
- 业界 survey:`docs/research/2026-05-09-w4a8-industry-kernel-survey.md`
- B7 baseline:`docs/experience/wins/2026-05-09-baseline-snapshot-d4c3fc3.md`(TTFT 1614 / ITL 23.2 / 90 tok/s c=4)

## 状态

QQQ 上游 3 分支调研完成。**Phase 1 = QQQ main bf16 activation(+21 LOC,低风险,
ITL -2-5%)** 是最佳起点。等 codex 当前 AWQ build 结束后,GPU 释放再 bench。
