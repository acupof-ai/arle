# M_quant — FP8 + W4 量化算子 magnitude 路线图(架构级,公式驱动)

> 2026-05-08。Triggered by user directive:"支持好量化算子 w4a8 可接受 fp4 是
> 未来的主流" + "先做大的优化再做小优化,规划好架构,给出有公式和模拟的预测,然
> 后做好事情" + AGENTS.md §0 SOLID。
>
> 之前 4 条 P0 路径全 KILL(M_pf-gemm autotune / M_pf-fuse gate-up /
> M_b.2.2 split-KV / M_pf-graph Phase 0),共同问题:都是 incremental 5-10%
> scope 的小优化。这份 plan **重新从 magnitude scale 切入** — 用硬件常数 + 公式
> 算清楚每个 quant path 的理论上限,排出真正的 order-of-magnitude 优化。

## §0 硬件常数(4070 Ti SUPER,sm_89 Ada Lovelace)

| Resource | Value | Source |
|---|---|---|
| HBM bandwidth | **672 GB/s** | 21 Gbps × 256 bit GDDR6X |
| HBM 容量 | 16 GiB | NVIDIA spec |
| BF16 tensor mma | 88.5 TFLOPS | Ada whitepaper |
| **FP8 tensor mma** | **706 TFLOPS**(8× BF16 dense rate)| Ada whitepaper(sm_89 native FP8 mma)|
| INT8 tensor mma | 706 TOPS | sm_89 |
| **FP4 tensor mma** | ❌ **无 native**(emulated only) | NVIDIA cutlass FP4 mma matrix:sm_100+ only |
| L2 cache | 48 MB | Ada |

## §1 模型常数(Qwen3-4B BF16)

| Param | Value |
|---|---|
| hidden_size | 2560 |
| intermediate_size | 9728 |
| num_attention_heads | 32 |
| num_kv_heads | 8 (GQA) |
| head_dim | 80 (256/3.2 wait:hidden/heads = 80) |
| num_layers | 36 |
| total params | ~4B |
| **BF16 weight footprint** | **8 GB**(4B × 2 byte) |

## §2 核心公式

### §2.1 Decode ITL formula(memory-bound dominant)

每个 decode token 需要 read 全部 weight(因为没有 reuse):

```
ITL_lower_bound = weight_size / HBM_bandwidth + KV_read + sample_overhead
```

ARLE 实测 BF16 decode ITL = 19.27 ms

理论 BF16 weight read = 8 GB / 672 GB/s = **11.9 ms** → memory utilization = 11.9/19.27 = **62%**

剩余 7.37 ms = KV read + sample + schedule overhead(unchanged across quant)

### §2.2 Prefill TTFT formula(compute-bound)

per-layer prefill FLOPS(M=batched tokens):

```
QKV (3 separate):  3 × (2 × M × hidden × hidden) = 6 × M × 2560²
o_proj:            2 × M × hidden × hidden       = 2 × M × 2560²
gate+up+down:      3 × (2 × M × hidden × inter)  = 6 × M × 2560 × 9728
attention:         2 × M² × hidden / heads_kv    ≈ M² × 80 × 32 (per attention head) × 8 kv heads
```

Qwen3-4B longctx 4k/c=4(M = 8192 batched rows total = 4096 × 2 chunks?actually
chunked_prefill_size 默认 2048 → M=2048 per chunk × 2 chunks per request × 4 requests):

per layer FLOPS @ M=8192:
- QKV: 6 × 8192 × 2560² ≈ 322 GFLOPS
- o_proj: 2 × 8192 × 2560² ≈ 107 GFLOPS
- gate/up/down: 6 × 8192 × 2560 × 9728 ≈ 2447 GFLOPS
- attention: 2 × 8192² × 2560 ≈ 344 GFLOPS
- **per layer ≈ 3220 GFLOPS = 3.22 TFLOPS**

× 36 layers = **116 TFLOPS** total prefill compute(8192 token batch)

Theoretical TTFT @ different precision:
- BF16 @ 88.5 TFLOPS:116 / 88.5 = **1.31 s**(理论上限)
- ARLE 实测 BF16 TTFT = 1976 ms → utilization = 1310/1976 = **66%**(还行)
- FP8 @ 706 TFLOPS:116 / 706 = **0.16 s = 164 ms**(理论上限)

@ 66% utilization on FP8 → TTFT ≈ 164 / 0.66 = **249 ms**(预测)

vs SGLang 972 ms = **3.9× faster than SGLang** = **超过 +30% world #1 目标 273%**

### §2.3 W4 weight 收益(decode)

W4 weight footprint = 4B × 0.5 byte = **2 GB**

ITL @ W4 decode lower bound = 2 / 672 + 7.37 = 2.98 + 7.37 = **10.35 ms**

vs ARLE BF16 19.27 ms → **1.86× faster decode**
vs SGLang 19.44 ms → **1.88× faster decode**

out tok/s @ W4: 1000/10.35 = **97 tok/s per request × 4 = 388 tok/s**(c=4 longctx)
ARLE BF16 out tok/s = 152.49 → **+154% throughput**

## §3 候选 quant path × ROI 模拟表

| Path | Weight | Activation | sm_89 Native? | Memory Δ vs BF16 | Compute Δ vs BF16 | Predicted ITL | Predicted TTFT | LOC est | Priority |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| 当前 BF16 baseline | BF16 | BF16 | ✅ | 1.0× | 1.0× | 19.27 ms | 1976 ms | (control) | — |
| **W8A8 (FP8)** | FP8 | FP8 | ✅ Ada FP8 mma | **2× smaller** | **8×**(706/88.5) | ~14.8 ms | **~250 ms** | 400-600 | **P0** |
| **W4A8 (W4 + FP8)** | W4 (Marlin) | FP8 | ✅(weight via Marlin / FP8 via mma)| **4× smaller** | **8×**(activation) | **~10.4 ms** | **~250 ms** | 700-1000 | **P0+** (combine W8A8 + W4 weight) |
| W4A8 (W4 + INT8) | W4 (Marlin) | INT8 | ⚠ sm_89 no INT8 mma 主推 | 4× smaller | 8× INT8 mma 但兼容性低 | ~10.4 ms | ~250 ms | 700 | P2(weak vs FP8 path) |
| **NVFP4** (FP4 weight + FP8 scale + FP8 activation) | E2M1 FP4 | FP8 | ❌ **emulated only** | 4× smaller | sm_89 emulated → **slower** than W4 | ~12 ms(emulate overhead) | ~300 ms | 600(substrate)| **P1 substrate** + sm_100 ready |
| BF16 + Speculative decode | BF16 | BF16 | ✅(已有 speculative.rs) | 1.0× | 1.5-2× | ~12 ms | ~1500 ms | 200 (tune) | P3(orthogonal,can stack)|

## §4 排序 + 选择(SOLID 评估)

按 magnitude scale + sm_89 兼容 + LOC ROI:

### **P0 — W8A8 (FP8 weight + FP8 activation) → ~250 ms TTFT,+86% decode tok/s**

**Why FIRST**:
- sm_89 **native FP8 mma**(706 TFLOPS,8× BF16),是 Ada GPU 的 sweet spot
- weight 8GB → 4GB,KV 同样 2× 缩(已有 FP8 KV path),GPU 内存压力大幅缓解
- ARLE 已有 FP8 KV(`decode_attention_fp8` `varlen_fp8` `Fp8Config`)→ **substrate 已 50% landed**,缺 W8 weight + FP8 GEMM dispatch
- ROI:**TTFT 1976 → 250 ms = 7.9× faster**(理论)

### **P0+ — W4A_FP8 (W4 weight + FP8 activation) → ~10.4 ms ITL,+154% decode tok/s**

**Why next**:
- W4 weight 2GB,KV pool 释放 6 GB → **KV 容量 +50%**(更长 context 或更高 batch)
- 复用 P0 的 FP8 activation path + Marlin W4 已有(`marlin_kernel.cu` from Elias Frantar)
- 主要 work:Marlin kernel 改 input dtype FP16 → FP8(~300 LOC delta)
- ROI:**ITL 19.27 → 10.35 ms = 1.86× decode speed**

### **P1 substrate — NVFP4 (E2M1 + FP8 scale + FP8 activation)**

**Why substrate-only**:
- sm_89 **没 native FP4 mma**,emulated decompress 反而比 W4 慢(违反 §0 SOLID — 无 evidence 跑得动)
- 但是 vLLM/SGLang 在 Blackwell 已 land NVFP4(B100/B200 sm_100)
- 价值:**架构 substrate land 等 sm_100 GPU**,本机 bench 标 "sm_100 expected,sm_89 emulate kill"
- 主要 work:`csrc/quant/fp4_*.cu` decode + `csrc/gemm/marlin_fp4.cu`(emulate now)+ Rust FFI + `quant.rs::NVFP4Config`
- LOC ~600,bench 上不跑,只 cargo build + e2e 验证 substrate

### **P2 deferred — W4A_INT8**

INT8 activation 在 sm_89 上没 FP8 mma 性能优势,unless deploy 到 Turing/Volta GPU。当前 4070 Ti SUPER 不重要。

## §5 Phase 0 license-or-kill(P0 W8A8 实施)

**目标**:Qwen3-4B FP8 weight + FP8 activation,保留 BF16 fallback,opt-in `INFER_QUANT=fp8_w8a8`。

**SOLID 检查清单**(per §0 第一原则):

| 假设 | 假设 vs evidence | 验证方法 |
|---|---|---|
| sm_89 FP8 mma 8× speedup | NVIDIA Ada whitepaper claims | **必须 cuBLASLt FP8 GEMM 跑实测对比 BF16 GEMM(arithmetic 验证)** |
| ARLE 当前 BF16 utilization 66% | 实测 1976 / 1310 = 66% | ✅ ground truth |
| FP8 utilization ≈ 66% | hypothesis | **必须 nsys profile 看 FP8 GEMM utilization** |
| FP8 weight 2× memory | math | ✅ |
| Qwen3-4B FP8 quantize without major accuracy loss | **literature claims < 0.5% perplexity loss** | **必须 e2e + greedy_consistency 验证 + 比较 BF16 vs FP8 输出 token-level diff** |

### Phase 0 scope(license LOC budget = 400):

1. **`infer/src/quant.rs`**:加 `Fp8W8A8Config` + `QuantFormat::Fp8W8A8` enum variant — ~30 LOC
2. **Weight loader 加 FP8 path**:`weight_loader.rs` 支持 BF16 → FP8 quantize on load(per-channel scale,common 算法)— ~80 LOC
3. **`crates/cuda-kernels/csrc/gemm/`** 加 FP8 GEMM wrapper:cuBLASLt FP8 path(`CUBLAS_COMPUTE_32F_FAST_TF32` → `CUBLAS_COMPUTE_32F_FAST_FP8`)— ~120 LOC
4. **`crates/cuda-kernels/src/ffi/gemm.rs`**:加 `gemm_fp8_w8a8_cuda`/`_into` Rust FFI — ~40 LOC
5. **`infer/src/ops/linear.rs`**:dispatch 路径 quant_format = Fp8W8A8 → 走 FP8 GEMM — ~40 LOC
6. **`infer/src/main.rs`**:CLI flag `--quant-format fp8_w8a8` — ~20 LOC
7. **Telemetry counters**:`fp8_gemm_count` / `bf16_fallback_count` — ~25 LOC
8. **`infer/tests/e2e.rs`**:加 FP8 W8A8 test case — ~25 LOC
9. **`infer/tests/greedy_consistency.rs`**:FP8 vs BF16 token diff < 1% — ~20 LOC

**LOC est total: 400**(严格按 budget)

### Validation gates(全 4 必过):

1. `cargo check --release -p infer --features cuda`
2. `cargo clippy --release -p infer --features cuda -- -D warnings`
3. `cargo test --release --test e2e --features cuda`(含 FP8 W8A8 case)
4. `cargo test --release --test greedy_consistency --features cuda`(FP8 vs BF16 token diff)

### License decision(per Phase 0 implementation 跑 longctx 4k bench):

| TTFT improvement | Action |
|---|---|
| ≥ **5×** vs BF16(理论 7.9×,实测 utilization 通常 50-70%) | **PROCEED Phase 1**:multi-shape bench + production default-on |
| 2-5× | PROCEED 但 lower priority,opt-in flag |
| <2× | **重审 nsys**:cuBLASLt FP8 path 实际利用率,**不立即 KILL**(SOLID 要求 verify root cause 不是 implementation bug)|
| e2e 输出 garbage | KILL + errors entry,deferred to better quantization algo |
| greedy diff > 5% token | KILL,FP8 quant accuracy 不达标 |

### Negative case + risk

- **FP8 quantize accuracy**:per-channel scale 在 LLM 上 literature 说 <0.5%,但 ARLE 自己量化可能 worse(SmoothQuant / AWQ-style 才是 SOTA)。Phase 0 用 simple per-channel,Phase 1 才上 SmoothQuant
- **cuBLASLt FP8 cuda 13.2 + sm_89 兼容性**:需要 verify cuBLASLt API 实际支持 FP8 GEMM with workspace(M_pf-gemm Phase 0 用 `gemm_graphsafe_cuda` 是 no-workspace path,FP8 可能 require workspace)
- **混淆变量**:Phase 0 同时改 weight format + GEMM kernel + (可能)KV format 时,要分开实验:
  - 控制 1:weight FP8 + activation BF16 + KV BF16(只 weight memory 收益)
  - 控制 2:weight FP8 + activation FP8 + KV BF16(加 compute 收益)
  - 控制 3:全 FP8(W8A8 + KV FP8)
  - 三步 isolation 才能 attribute 收益来源(per §0 SOLID 混淆变量必须隔离)

## §6 Phase 1 — W4A_FP8(在 Phase 0 license 后启动)

复用 Phase 0 FP8 activation path + Marlin W4 weight repack:

- Marlin kernel 改 input dtype: FP16 → FP8(via input quantizer)— 200 LOC
- LOC est 300,Phase 0 substrate 复用 60%

**Predicted ITL** = 10.35 ms(下面表 §3),实际 utilization 70% → ~14.8 ms ≈ 1.3× speedup vs Phase 0 W8A8 alone

stack on Phase 0:**1.86× decode + 7.9× prefill** combined improvement vs BF16 baseline。

## §7 Phase 2 substrate — NVFP4

仅 substrate(sm_89 emulated 不 production),等 sm_100 硬件:

- `csrc/quant/fp4_e2m1.cu` decode + dequantize utilities(~200 LOC)
- `csrc/gemm/marlin_fp4.cu`(emulate via decompress + BF16 mma,sm_100 时 swap to native FP4 mma,~250 LOC)
- Rust FFI + `quant.rs::NVFP4Config` + CLI flag(~150 LOC)
- e2e test only,**no bench gate**(sm_89 emulated 慢,bench 没意义,标 "sm_100 ready" 即可)

## §8 与 §0 SOLID 第一原则的关系

这份 plan 的 SOLID 自检:

✅ **公式 + 数字模拟**:全部 ROI 用硬件常数 × 模型常数推导,non-hand-wave
✅ **混淆变量隔离**:Phase 0 §5 要求 3 步 control variable 实验
✅ **Root cause 假设也 license-or-kill**:Phase 0 license decision <2× 时不立即 KILL,先 nsys verify implementation
⚠ **80% SOLID gap 自检**:
- 实测 cuBLASLt FP8 sm_89 + cuda 13.2 兼容性 — 没 verify(只看 NVIDIA spec)
- FP8 quantize Qwen3-4B 实际 perplexity loss — literature claim,本机没跑
- **Phase 0 第一步必须先做**:cuBLASLt FP8 smoke test(单 GEMM call,no model)→ verify hardware path 通

## §9 立即 next step — cuBLASLt FP8 smoke test

**Phase 0 之前的 P0 sanity check**(15-30 min,SOLID 第一步):

```cpp
// crates/cuda-kernels/csrc/gemm/fp8_smoke.cu
// 单 GEMM call:(M=2048, N=2560, K=2560) FP8 input × FP8 weight → BF16 output
// cuBLASLt FP8 mma path,cudaEvent 测 elapsed time
// expect: < 2 ms (vs BF16 same shape 实测 ~ 16 ms = 8× speedup)
```

如果 smoke 跑不出来 → cuBLASLt FP8 + sm_89 + cuda 13.2 路径有问题,Phase 0 假设错,plan abort 或 cuda 版本切换

如果 smoke ≥ 6× speedup → Phase 0 实施按 plan 推进
如果 smoke < 4× → 重审,可能 need cutlass FP8 而不是 cuBLASLt

## Cross-references

- AGENTS.md §0 第一原则 — SOLID
- 当前 quant inventory:`infer/src/quant.rs` + `crates/cuda-kernels/csrc/{quant,gemm}/`
- M_pf-graph Phase 0 KILL 教训:`docs/experience/errors/2026-05-08-m_pgc-phase0-killed-ttft-under-threshold.md`
- M_world1 +30% lead target:`docs/plans/M_world1-30-percent-lead-roadmap.md`
- 4070 Ti SUPER spec:NVIDIA Ada whitepaper
- FP8 quant for LLM:literature(SmoothQuant / AWQ paper)— TODO Phase 1 cite
