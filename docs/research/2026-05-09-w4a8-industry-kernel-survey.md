# W4A8 业界 kernel survey + ARLE 升级路径(`d4c3fc3` baseline 锚点)

> 用户问 "业界 W4 quant 怎么写得这么快,能抄哪个" → 调研 5 个开源 W4 quant
> kernel 实现,确定 ARLE sm_89 RTX 4070 Ti SUPER 上**可抄的 + 不能抄的**。
>
> **结论 TL;DR**:Machete = Hopper-only(无法抄)。**可抄 = vLLM 当前
> 演化版的 Marlin**(我们用的是 PR #31 老 cherry-pick,vLLM 已升级到
> 5000 LOC 多 specialization 版本)。

## 当前 ARLE W4A8 状态(baseline `d4c3fc3`)

- **Kernel**:`crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu`(987 LOC,单文件)
- **来源**:PR #31 cherry-pick(`a019a0e feat(cuda): cherry-pick PR #31 W4A8 marlin kernel + ARLE adapter`)
- **每次 linear call**:3 launches(quantize → marlin gemm → fp16→bf16)+ 5 alloc_zeros
- **B7 bench**:TTFT 1614ms,ITL 23.2ms,90 tok/s,σ-tight(per `2026-05-09-baseline-snapshot-d4c3fc3.md`)

## 业界 W4/W4A8 quant kernel 全览

### 1. ❌ Machete(vLLM)— **Hopper-only,sm_89 不能用**

- `vllm-project/vllm:csrc/quantization/machete/`(~3000 LOC)
- Spiritual successor to Marlin,**基于 Cutlass + WGMMA + TMA**
- 明确 `using ArchTag = cutlass::arch::Sm90`(只支持 sm_90+)
- **结论:sm_89 RTX 4070 Ti SUPER 完全不能用**,跳过

### 2. ✅ **vLLM 当前 Marlin**(PR #31 后续演化版)— **可抄首选**

- `vllm-project/vllm:csrc/quantization/marlin/`(~5000 LOC across 6 files)
- License:Apache 2.0(可直接 cherry-pick)
- `marlin_gemm(..) requires CUDA_ARCH >= 7.5` → **sm_75/80/89 都支持**

文件结构:
| 文件 | LOC | 作用 |
|------|----:|------|
| `marlin.cu` | 863 | 主 dispatch,Python/C++ FFI |
| `marlin_template.h` | **2081** | **多 N/K shape specialized kernel templates** ← 性能关键 |
| `dequant.h` | 609 | **重构后的 4-bit unpack**(更优 PTX intrinsic) |
| `marlin_mma.h` | 268 | mma 指令 wrapper |
| `marlin_dtypes.cuh` | 149 | scalar_type 抽象(half/bf16/fp8 通用) |
| `marlin_int4_fp8_preprocess.cu` | 106 | **新 W4 + FP8 activation 路径**(sm_89 native FP8!) |
| `gptq_marlin_repack.cu` | 357 | GPTQ → marlin layout repack |
| `awq_marlin_repack.cu` | 288 | AWQ → marlin layout repack |

vs 我们 PR #31 的关键差异:
- **多 dispatch shape**:N=2560 / 5120 / 13824 / ... 各有特化 kernel(我们只一个 path)
- **dequant.h 重构**:更优的 4-bit unpack PTX(可能 ITL -3-8% if hot loop)
- **`is_k_full` mode**:K 是否 group_size 整数倍优化路径
- **`use_atomic_add` reduce**:避免 reduce buffer alloc(我们当前 alloc max_par × 64 × n!)
- **`use_fp32_reduce`**:精度可调(BF16 worst-case bias 修正)
- **`b_bias_or_none`**:linear with bias 直接 fuse
- **GPTQ `g_idx + perm`**:act-order 模式(我们 zpfix 已绕过)
- **AWQ marlin repack**:支持 AWQ checkpoint 直接 load

### 3. ⚠ AWQ kernel(MIT Han Lab)

- `mit-han-lab/llm-awq:awq/kernels/csrc/`
- License:MIT
- **量化方法 + kernel**:salient channel 保留 FP16,其余 W4
- **优势**:accuracy 比 GPTQ 高(per AWQ paper)
- **劣势**:per-token 处理 outlier 有少量 overhead
- **是否抄**:AWQ 量化方法可学,但 kernel perf 不显著优于 vLLM marlin

### 4. ⚠ exllamav2(turboderp)

- `turboderp/exllamav2:exllamav2_ext/cuda/q_gemm.cu`
- License:MIT
- **特点**:纯 CUDA C(不依赖 cutlass),sm_70+ 通用
- **EXL2 quant format**:variable bit-width per channel
- **不直接抄**:format 不一致,但可学 dispatch 思路

### 5. ❌ TensorRT-LLM W4A8

- NVIDIA closed-source
- 已知性能:Qwen2-7B W4A8 sm_89 ~245 tok/s c=1
- **不能抄**(闭源)
- **作 upper bound 锚点**:我们 B7 c=4 90 tok/s,c=1 推算 ~50 tok/s × ITL 23ms = upper bound 是 50 → ~245 是 **有 5× 提升空间**(他们高度优化)

## ARLE 升级路径推荐(按 ROI 排序)

### 🥇 **P0 推荐 — 移植 vLLM 当前 marlin(分阶段)**

**Phase 1**(Claude 1-2 天 OR codex 1 天):**抄 dequant.h + atomic_add option**
- LOC:~700(dequant.h 609 + adapter 调整)
- 风险:低(纯 lookup 重构 + 减 alloc)
- 预估 gain:**ITL -3-8%(dequant 更优)+ TTFT -2-5%(reduce buffer 省)**
- 验证:greedy_consistency + B7 bench Δ%

**Phase 2**(codex 2-3 天):**多 N/K shape specialization**
- LOC:~2000(marlin_template.h 多 instantiate)
- 风险:中(需精确 dispatch table)
- 预估 gain:**ITL -5-15%**(N=2560 Qwen3-4B 特化 + N=5120 attention out_proj)
- 验证:greedy_consistency + 全 baseline matrix re-bench

**Phase 3**(codex 1 天):**FP8 activation path**(`marlin_int4_fp8_preprocess.cu`)
- LOC:~300
- 风险:中(sm_89 native FP8 mma 路径)
- 预估 gain:**ITL -10-25%**(FP8 mma 706 TFLOPS vs BF16 88.5 = 8× peak,但 quant 实际 ~1.5-2.5× decode)
- 验证:vs B7 baseline + greedy

### 🥈 **P1 平行轴 — 完成 #24 W4A8 graph capture hoist**(已在队列)

不依赖 marlin 升级,independently can land。预估 TTFT -5-15%。

### 🥉 **P2 长期 — sm_89 specific tile re-tune**(skill anti-pattern #4)

ncu profile + BLOCK_M / NUM_STAGES sweep,等 ncu wrapper migration 完成后再做。

## 不推荐(或低优先)

- ❌ Machete:Hopper-only
- ⚠ AWQ:量化方法可学,kernel 不显著快
- ⚠ exllamav2:format 不兼容
- ❌ TensorRT-LLM:闭源

## 立即 next step(本 session)

如果用户 OK,我**先开始 Phase 1**(抄 vLLM 当前 dequant.h + atomic_add reduce):
1. Fetch vllm-project/vllm 的最新 marlin
2. Diff dequant.h vs 我们 PR #31 的 unpack 代码
3. Port + adapter 调整
4. cargo check + greedy_consistency
5. B7 bench Δ% vs `d4c3fc3` baseline
6. License-or-kill 决策

如果 Phase 1 PASS,继续 Phase 2(多 shape spec)。

## Cross-references

- 当前 baseline:`docs/experience/wins/2026-05-09-baseline-snapshot-d4c3fc3.md`
- W4A8 当前 kernel:`crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu`(987 LOC)
- W4A8 adapter:`infer/src/ops/linear.rs:1307 run_marlin_w4a8_linear`(120 LOC)
- vLLM marlin upstream:https://github.com/vllm-project/vllm/tree/main/csrc/quantization/marlin
- vLLM machete(Hopper-only,跳过):https://github.com/vllm-project/vllm/tree/main/csrc/quantization/machete
- Skill anti-pattern #4:Hopper-default kernels on Ada(我们 marlin tile 也可能踩这个)

## 状态

业界 survey 完成。**Machete 不可用**(Hopper-only),**vLLM 当前 marlin = 升级首选**。
分 3 phase 移植 路径已规划。等用户 GO 即可开始 Phase 1。
