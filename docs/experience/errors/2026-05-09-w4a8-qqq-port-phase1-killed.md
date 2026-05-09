# W4A8 QQQ main port Phase 1 — KILLED via source-level audit(no actual port)

> 接续 `2026-05-09-w4a8-qqq-real-diff-finding.md`(QQQ main +119 LOC delta
> 是 thread_config dispatch)。深入读 QQQ main 实际 dispatch 逻辑,**发现
> ARLE 当前已经命中 Qwen3-4B 实际 thread_config**,Phase 1 port 对
> Qwen3-4B 估计 perf gain **<5%,不值得 +130-150 LOC + 编译时间**。
> KILL hypothesis,转其他 axis。

## Phase 0 audit(skill rule 3 binding constraint identification)

### ARLE 当前 marlin_w4a8_kernel.cu 实际 dispatch

```cpp
// ARLE current (line 888-897)
if (thread_k == -1 || thread_n == -1) {
  if (prob_m <= 16) {
    thread_k = 128;  // small batch (decode/c=4)
    thread_n = 128;
  } else {
    thread_k = 64;   // large batch (prefill seq>16)
    thread_n = 256;
  }
}
```

ARLE 已经有 small/large batch dispatch **已是 QQQ main same default!**
唯一缺的是 **fallback configs**(QQQ main 有 4 个 fallback per batch class):

| | ARLE 当前 | QQQ main |
|---|---|---|
| small_batch default | (128, 128) | (128, 128) **same** |
| small_batch fallback | 无 | 3 个(128,64)(64,256)(64,128) |
| large_batch default | (64, 256) | (64, 256) **same** |
| large_batch fallback | 无 | 3 个(128,128)(64,128)(128,64) |

### Qwen3-4B 实际 dispatch 命中

Qwen3-4B 形状:
- `hidden_size = 2560`
- `intermediate_size = 8960`
- `q_proj/k_proj/v_proj/o_proj`:`prob_n` = 2560(out) `prob_k` = 2560(in)
- `gate_proj/up_proj`:`prob_n` = 8960 `prob_k` = 2560
- `down_proj`:`prob_n` = 2560 `prob_k` = 8960

所有 layer:**`prob_n % 128 == 0` 且 `prob_k % 128 == 0`** ✓ small_batch default `(128, 128)` 直接命中
**`prob_n % 256 == 0` 且 `prob_k % 64 == 0`** ✓ large_batch default `(64, 256)` 直接命中

→ **fallback configs 完全不会 fire**(除非未来加 hidden=384/768/etc 不 divisible 模型)

### 真实 Phase 1 port gain estimate

| 维度 | ARLE 当前 | QQQ main 替代 | Qwen3-4B 实际差异 |
|------|----------|---------------|------------------|
| dispatch tile | hardcoded (128,128)/(64,256) | dynamic + fallback | **0%(命中相同 default)** |
| num_threads | hardcoded 256 | 选 128 OR 256 per config | small batch `num_threads=128` 可能 occupancy +1-3% |
| group_blocks | enumerate -1, 8 | 同 | **0%** |
| L2 cache hint | ✓ ARLE 已有 | ✗ QQQ main 无 | **ARLE +优势**(已 ahead) |
| compile time | 10 cubin | ~40 cubin | -- |
| binary size | -- | +~30 KB | -- |

→ **Phase 1 perf gain 估计 < 5% on Qwen3-4B**,且付出 +130-150 LOC + ~30 cubin × 编译时间 + binary size 增加。**ROI 不够 license**。

## KILL 决策(per skill rule 8 license-or-kill)

**License threshold**:典型 quant kernel optimization 需 ITL Δ ≥ 5-10% 才值得 substrate change。
**Predicted Δ**:< 5%(per Phase 0 audit 上)
**KILL signal**:✗(预估低于 license,且 ARLE 已 ahead in L2 cache hint)

→ **W4A8 QQQ main port Phase 1 KILLED at audit stage**,**不 port,不 bench**。

## 业界真正的 W4A8 优化方向(survey 回顾)

回头看完整 survey,真正能给 W4A8 perf gain 的轴:

| 轴 | 来源 | 预估 gain | 可行性 |
|----|------|----------|--------|
| **fp16→bf16 fuse 进 kernel epilogue** | 100-200 LOC kernel BF16-ization | ITL -3-8% | 中(数值精度需验证) |
| **scratch buffer hoist 全 prefill 路径** | #24 W4A8 graph capture hoist (in queue) | TTFT -5-15% | 已在 codex queue |
| **sm_89 specific tile re-tune**(skill #4) | ncu profile + BLOCK_M sweep | ITL -5-15% | profiler-blocked(ncu wrapper migration) |
| **QQQ i4fp8 FP8 activation path** | +167 LOC + adapter | accuracy ↑,perf ~0% | 中(需 FP8 quant 路径) |
| **W4A8 graph capture hoist (#24)** | 200-400 LOC substrate | TTFT -5-15% | codex queue 待 pickup |

**最高 ROI now**:
1. **`#24 W4A8 graph capture hoist`** — 已在 codex queue,等 codex 执行
2. **fp16→bf16 fuse**(可 Claude 做,但 100-200 LOC kernel-internal 改动需精度验证)
3. **scratch hoist 完成全 prefill 路径**

## Phase 0 audit ROI

**0 LOC + 1 hour Claude wall-clock 节省**:
- 没去 +130-150 LOC + ~30 cubin extra + 编译时间 + binary size 的 Phase 1 port
- 没去 build + bench 失败的 Phase 1 verify(估计 30-60 min wall-clock)
- 没去 KILL entry 复盘(估计 30 min)

**总 saving:~1.5-2 hours wall-clock + 100-150 LOC churn**(per skill v1.7.0 #18 Phase 0 substrate audit)。

## 战略决定

W4A8 marlin axis 的 **kernel-level 改动收益已饱和** on Qwen3-4B sm_89:
- ARLE 已有 sm_89 L2 cache hint(vLLM team 加的)
- ARLE 已有 small/large batch tile dispatch default(命中 Qwen3-4B 最优)
- QQQ main thread_config dispatch 主要是 robustness(其他模型),不是 perf

**真正瓶颈在 adapter 层**(scratch alloc + launch overhead),已部分 codex P1.2 ca0673b 和 #24 graph capture hoist 在解决。

**结论**:停 W4A8 kernel-internal 优化轴。pickup queue 已有 #24,**等 codex 执行 + 关注 scratch hoist 完成度**。Claude 转其他 axis。

## Cross-references

- 之前调研误报:`2026-05-09-w4a8-qqq-real-diff-finding.md`(QQQ main +119 LOC,但实际 Qwen3-4B 不 fire)
- 业界 survey:`2026-05-09-w4a8-industry-kernel-survey.md`
- 业界纠正:`2026-05-09-w4a8-upstream-qqq-survey.md`
- ARLE current:`crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu:888-897`
- pickup queue #24:`docs/plans/codex-pickup-queue-2026-05-09.md`
- skill v1.7.0 anti-pattern #18:Phase 0 substrate audit before scoping new wiring

## Rule

**Source-level Phase 0 audit 是最便宜的 license-or-kill — 1 hour Claude
work 救了 1.5-2 hours wasted port + bench**。

QQQ main 的 thread_config dispatch 看起来 fancy,但深入读 ARLE 当前
default + Qwen3-4B 实际 prob_n/prob_k % 128 == 0 → 已经命中 same default。
**移植 = 0 perf gain 在 our hot path**。

教训:**业界 kernel "新版本" 不一定是 perf upgrade,可能是 robustness
扩展(更多模型支持)**。Phase 0 audit 必须看 default + fallback fire
condition,不能只看 LOC delta。

## 状态

W4A8 QQQ port Phase 1 **KILLED at audit**(0 LOC change)。Claude 转
其他 axis(可能:fp16→bf16 fuse OR 等 codex #24 完成后再决定)。
