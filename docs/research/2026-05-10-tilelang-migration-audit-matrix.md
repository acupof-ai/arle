# TileLang migration audit matrix — full ops inventory + sequenced plan

> 2026-05-10 EOD+187 — 用户 direction:"多看看用 tilelang,有不适合的地方可以提
> pr,希望能全部迁移或者适配 包括 w4a8 算子"。Spawn Explore audit produced
> systematic ops inventory + 4-tier feasibility matrix + 5 upstream PR
> opportunities + sequenced 2026 H1-H2 roadmap。
>
> §0 SOLID:**Marlin W4 hand-tuned tensor-core asm 可否 TileLang code-gen
> parity 是 open question** — 需 micro-benchmark gate before commit。

## Executive summary

ARLE CUDA kernel 层当前 **52% custom CUDA + 48% library/TileLang**。**~14.5K
LOC custom CUDA C(38 files)**,其中 **~6-8K LOC 可即 migrate 到 TileLang**
without hand-tuned asm replacement。

战略 bottom line:
- **Tier 2 immediate(~850 LOC,2 weeks,zero upstream gate)** ✅ 可立即 ship
- **Tier 3 medium(~1.5K LOC,upstream coord)** ⚠ 需 FP8 scale semantics PR
- **Tier 4 high-risk(~1.8K LOC W4 Marlin)** 🔴 hand-tuned asm,**no proven
  parity** with TileLang SASS code-gen → 需 benchmark gate

## Current TileLang ops inventory(grep-verified)

| Op | FFI prefix | Cubin variants | Caller | LOC(总)|
|----|-----------|---------------|--------|---------|
| Paged Prefill HD128 BF16 | `tilelang_batch_prefill_paged_hd128_*` | 4 configs(q16/32/40/64_kv8)| `infer/src/ops/attention.rs:prefill_paged_hd128()` | ~4K py |
| Paged Prefill HD256 BF16 | `tilelang_batch_prefill_paged_hd256_*` | 3 configs(q8/16_kv2/4)| `infer/src/ops/attention.rs:prefill_paged_hd256()` | ~3K py |
| Paged Prefill HD64 BF16 | `tilelang_batch_prefill_paged_hd64_*` | 1 config | `infer/src/ops/attention.rs:prefill_paged_hd64()` | ~2.5K py |
| Paged Decode HD128 BF16 | `tilelang_batch_decode_paged_hd128_*` | 4 configs | `decode_paged_hd128_full()` | ~7K py(complex split-KV)|
| Paged Decode HD256 BF16 | `tilelang_batch_decode_paged_hd256_*` | 3 configs | `decode_paged_hd256()` | ~3.5K py |
| Paged Decode HD64 BF16 | `tilelang_batch_decode_paged_hd64_*` | 1 config | `decode_paged_hd64()` | ~2.5K py |
| Paged Decode HD128 FP8 | `tilelang_batch_decode_paged_hd128_fp8_*` | 1 config(q32_kv8,**built but unwired**)| FFI declared `crates/cuda-kernels/src/ffi/attention.rs:735` | ~2K py |
| GDR(Gated Delta Rule)| TileLang + native CUDA hybrid | Chunk-wise prep + strict-LT solve | `infer/src/ops/recurrent.rs` | ~5K py + 83 C |

**关键**:TileLang 仅覆盖 **paged attention + GDR chunk prep**;**无 linear /
norm / elementwise / sampling**。

## Non-TileLang ops inventory(14.5K LOC,38 .cu files)

| Category | Op | File | Kernel type | LOC | Tensor-core? |
|---|---|---|---|---|---|
| **Linear/GEMM** | BF16 GEMV(decode) | `csrc/gemm/gemv.cu` | custom CUDA vectorized | 741 | No |
|  | BF16 GEMM(prefill) | cuBLAS wrapper | library | 0 | Yes(cuBLAS)|
|  | W4A16 GEMV(decode) | `csrc/gemm/quantized_gemv.cu` | custom CUDA | 1672 | No |
|  | **W4A16 Marlin(prefill)** | `csrc/gemm/marlin_kernel.cu` | **hand-tuned tensor-core asm** | **844** | **Yes hand-tuned** |
|  | **W4A8 Marlin** | `csrc/gemm/marlin_w4a8_kernel.cu` | **hand-tuned tensor-core asm** | **987** | **Yes hand-tuned** |
|  | W8A16/W2A16 GEMV | `csrc/gemm/quantized_gemv.cu` | custom CUDA | 1672(shared)| No |
|  | TurboQuant GEMV | `csrc/gemm/turboquant_weight_gemv.cu` | custom dequant | 271 | No |
|  | Q3K/Q4K/Q5K/Q6K GEMV | `csrc/gemm/quantized_gemv.cu` | custom CUDA | 1672(shared)| No |
| **Norm** | RMSNorm | `csrc/misc/norm.cu` | custom parallel reduce | **1052** | No |
|  | Fused add+RMSNorm | `csrc/misc/norm.cu` | custom CUDA | 1052(shared)| No |
| **Elementwise** | SiLU+mul / Add / split_qkv | `csrc/misc/elementwise_basic.cu` | custom pointwise | **210** | No |
| **Attention prep** | Prefill RoPE+norm | `csrc/attention/prefill_attention_*.cu` | custom CUDA | 331 | No |
|  | Decode RoPE+norm | `csrc/attention/decode_prep_paged*.cu` | custom CUDA | 577 | No |
|  | Non-paged attention | `csrc/attention/nonpaged_prefill_attention.cu` | custom CUDA | 158 | No |
| **KV quantization** | TurboQuant KV quant | `csrc/quant/turboquant.cu` | custom Lloyd-max | 825 | No |
|  | KV dtype convert | `csrc/quant/dtype_convert.cu` | custom CUDA | 53 | No |
|  | Paged KV metadata + append + pool→paged | `csrc/kv/*.cu` | custom CUDA | 518 | No |
| **Sampling** | argmax / top-k / nucleus | `csrc/misc/sampling.cu` | custom parallel scan | **632** | No |
| **Recurrent** | Conv1D | `csrc/misc/conv1d*.cu` | custom CUDA | 291 | No |
|  | GDR solve | `csrc/misc/gdr_*.cu` | custom strict-LT solve | 766 | No |

## Migration feasibility matrix

| Op | Current | TileLang feasibility | Upstream gap? | Tier | Est. LOC | Risk |
|----|---------|---------------------|---------------|------|----------|------|
| BF16 linear prefill | cuBLAS | High(matmul DSL native)| None | 2 | ~200 | Low |
| BF16 GEMV decode | custom | Medium | None | 2 | ~150 | Low |
| W4A16 Marlin prefill | asm 844 | Medium(quant partial)| GPTQ pack support? | 3 | ~400-600 | Medium |
| **W4A16/W4A8 Marlin** | **asm 1831** | **Low**(hand-tuned tensor-core mma) | **W4A8 DSL upstream gap** | **4** | **~600-1000** | **HIGH** |
| W8A16 / W2A16 / GGUF | custom | Medium(quant DSL exists) | Partial | 2-3 | ~300-400 | Low-Med |
| TurboQuant GEMV | custom | Medium(stateless dequant)| None | 2 | ~200 | Low |
| RMSNorm | custom 1052 | High(reduction DSL)| None | 2 | ~100-150 | Low |
| **SiLU / add / split_qkv** | custom 210 | **Very High**(pointwise sweet spot)| None | **1** | ~50 | **Minimal** |
| Prefill / Decode RoPE+norm | custom 908 | High(fused pointwise+reduce)| Minor schedule fusion | 2 | ~150 | Low |
| **FP8 KV decode** | custom 431 | Medium(FP8 dequant)| **Major: scale layout PR** | 3 | ~300-400 | Medium-High |
| Sampling | custom 632 | Medium(scan-based)| None | 2 | ~200 | Low |
| Conv1D | custom 291 | **Low**(no DSL support)| Major upstream | 4 | ~400-600 | High |
| GDR | hybrid TileLang+C | Already hybrid ✓ | None | 1 | 0 | N/A |

## Upstream TileLang PR opportunities

| Rank | PR scope | Effort | Blocks |
|------|----------|--------|--------|
| **P0** | **FP8 KV-cache scale semantics** + tests | ~500 LOC + docs | Tier 3 FP8 KV decode(P1.4 KILL `51dd5b2` root cause)|
| **P0** | **W4A8 quantized matmul DSL**(INT8 act + INT4 weight + scales)| ~1-2K LOC | Tier 4 W4A8 Marlin |
| **P1** | GPTQ packed weight format support | ~800 LOC | Tier 3 W4A16 prefill optimize |
| **P1** | Pointwise op catalog(RMSNorm / SiLU / add / split_qkv)+ tests | ~500 LOC | Tier 2 elementwise/norm acceleration |
| **P2** | Scan/reduce primatives documentation | ~100 LOC docs | Tier 2 sampling |

## Sequenced migration roadmap

```
May 2026:
├─ Tier 2 prep(elementwise → RMSNorm → RoPE → sampling → BF16 GEMV)
│  └─ ~850 LOC eliminated,1-2 weeks,zero upstream blockers
├─ Engage upstream TileLang:FP8 semantics + W4A8 DSL discussion
│  └─ Parallel,inform Tier 3/4 scope

Jun-Jul 2026:
├─ Tier 3(quant unification) — 2-3 weeks if upstream unblocks
│  ├─ W8A16/GGUF via TileLang quant DSL
│  ├─ FP8 KV decode(if scale semantics resolved)
│  └─ Evaluate W4A16 prefill TileLang vs Marlin asm parity bench
├─ Upstream PRs(P0+P1)— parallel,~4 weeks

Aug-Sep 2026:
├─ Tier 4 decision gate
│  ├─ IF upstream W4A8 DSL ready: W4A8 Marlin migration planning
│  └─ ELSE: keep Marlin asm,document strategic exception
├─ Full regression suite + performance parity validation

Target outcome(EOY 2026):
├─ BF16 + elem + norm + sampling 100% TileLang
├─ Quantized linear(W8/W2/GGUF)~80% TileLang
├─ W4A16/W4A8 Marlin: keep hand-tuned asm if code-gen parity gap > 10%
├─ Conv1D + GDR solve: stay custom CUDA(no DSL support)
└─ ~6-8K LOC eliminated,ops layer unified on TileLang dispatcher
```

## Key insights + open questions

### Verified facts
- TileLang DSL semantically complete for matmul / gemv / pointwise / reduce / scan
- TileLang **NOT supported**:conv,inline tensor-core asm,W4A8 dequant DSL
- Marlin asm 是 hand-tuned tensor-core mma + async-copy hints — **TileLang SASS code-gen parity 未 proven**
- FP8 KV decode TileLang cubin 已 built but wire 失败(`51dd5b2` P1.4 KILL,scale layout 不对齐)

### Open questions(require micro-benchmark)
1. **Tensor-core parity**:TileLang SASS backend can produce mma utilization 等价于 Marlin asm? Target ≥90% throughput, micro-bench `T.gemm(m=128,n=256,k=4096,quant=w4)` vs hand-tuned Marlin on RTX 4090 / A100
2. **FP8 scale layout**:TileLang FP8 dequant assumes monolithic scale tensor or per-token broadcast?Upstream clarification needed
3. **Conv1D**:On TileLang 0.1.10+ roadmap?If not,Conv1D stays custom CUDA forever

## Risk + mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| TileLang SASS < Marlin throughput W4A16/W4A8 | High | Cannot migrate hand-tuned asm,1.8K LOC stays custom | Micro-bench early(Jun 2026);accept hybrid if gap > 10% |
| Upstream FP8 scale semantics not aligned | Medium | FP8 KV decode stays custom,Tier 3 blocked | Engage upstream now(May);workaround = scale reformat ARLE-side if needed |
| TileLang version pinning blocks W4A8 DSL | Low | W4A8 migration deferred 2027 | OK,W4A16 higher priority,W4A8 secondary |
| Regression in pointwise op latency Tier 2 | Very Low | Rollback simple | Comprehensive latency suite;gate <5% regression |

## Recommended next action

**Tier 2 dispatch**(when codex finishes #24 prefill graph hoist):
- Single elementwise op pilot first(SiLU+mul ~50 LOC)
- Validate substrate flow:TileLang DSL 编写 → AOT cubin build → FFI declare → Rust dispatch → numerical equivalence test → bench A/B vs custom CUDA
- 若 pilot 顺利:批量迁 RMSNorm + RoPE + sampling + BF16 GEMV

**Parallel**:开 upstream TileLang issue/discussion 关于 FP8 scale 语义(P0)+ W4A8 DSL(P0)— 这些 unblocks Tier 3+4 future work。

## Cross-references

- `51dd5b2` P1.4 TileLang FP8 decode wire KILL(scale 语义不对齐 evidence)
- `2778dc8` anti-pattern #26 candidate(same-output-but-garbage P1.4 evidence)
- `0969480` P0.0 Phase 1.B SGLang re-verify(ARLE 4k prefill +76.6% lag motivates faster prefill compute)
- `crates/cuda-kernels/csrc/` 38 .cu files inventory base
- `crates/cuda-kernels/tools/tilelang/` 或 similar — TileLang DSL source location
- `crates/cuda-kernels/src/ffi/attention.rs` — current TileLang FFI declarations
- TileLang upstream:`tile-lang/tilelang` GitHub
- §0 SOLID:**没 evidence 不下结论;tensor-core parity 必须 micro-benchmark before W4 migration commit**
