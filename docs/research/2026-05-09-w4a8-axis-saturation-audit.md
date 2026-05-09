# W4A8 axis 饱和 audit — kernel + adapter 改动收益已 < 5%(Qwen3-4B sm_89)

> 接续 `2026-05-09-w4a8-qqq-port-phase1-killed.md`(Phase 1 KILLED at audit)。
> Phase 0 source-level audit 全 W4A8 优化候选 axis,**结论**:kernel
> internal + adapter scratch hoist 都已饱和 < 5% gain on Qwen3-4B sm_89。
> 真正后续 axis 在 codex #24 graph capture hoist 或 完全不同 axis(spec
> decode / chunked prefill)。

## 5 个候选 axis Phase 0 audit 矩阵

| # | axis | LOC | 风险 | 实测/估计 gain | 状态 |
|---|------|-----|------|-----------------|------|
| 1 | **QQQ main thread_config dispatch port** | +130-150 | 中 | < 5%(Qwen3-4B 已命中 same default)| ❌ KILLED at audit `f2bde6b` |
| 2 | **fp16→bf16 fuse 进 kernel epilogue** | 100-200 LOC kernel rewrite | 中(精度) | ITL ~3-8%(if 实现) | ❌ KILLED at audit(本 brief)— Marlin kernel 全 hardcode FP16,不是 template scalar_t,需大 rewrite + accuracy verify,1+ day codex |
| 3 | **Prefill scratch hoist**(类似 decode `MarlinDecodeScratch`)| 200-300 LOC | 中 | TTFT ~2-5% | ❌ KILLED at audit(本 brief)— prefill 是 GEMM compute bound(nsys 97%),not alloc bound;LOC 多 gain 少 |
| 4 | **#24 W4A8 graph capture hoist**(codex queue)| 200-400 | 中 | TTFT -5-15% | 🟡 等 codex pickup |
| 5 | **sm_89 specific tile re-tune**(skill #4)| ncu sweep | 低 | ITL -5-15% | 🟡 ncu wrapper migration blocked |

## Phase 0 audit 详细发现

### Axis #2:fp16→bf16 fuse(KILL)

`marlin_w4a8_kernel.cu` 全 hardcode FP16:
- `FragS_GROUP = Vec<half2, 1>`(权重 group scales 是 FP16)
- `dequant_per_group` 内部全 `half2` ops + magic numbers `0x64806480` FP16 specific
- `((half2*) sh)[idx] = float2_to_half2(deq_res);` epilogue 是 FP16 store
- **不是 template `<scalar_t>`,是 hardcode `half`**

要 BF16-ization:
1. 所有 `half2` → `nv_bfloat162` OR `template <typename T2>`
2. dequant_per_group 重写(FP16 magic numbers → BF16 等价)
3. `float2_to_half2` → `__float22bfloat162_rn`
4. 数值精度验证(BF16 mantissa 7-bit vs FP16 10-bit,可能 PPL 退化)

**LOC**:100-200 + 数值精度调试(可能 1+ day codex work)
**Gain**:理论 ITL -3-8%(省 1 launch fp16→bf16 转换 per linear call × 252 calls/token)
**Risk**:中(BF16 mantissa 不足可能引入 PPL 退化)

→ KILL:LOC 大 + 风险中 + gain 中。等 ncu 实测 launch overhead 占比再决定值不值得 attempt。

### Axis #3:Prefill scratch hoist(KILL)

`linear.rs:1872`:`try_gemm_with_phase_into` 写死传 `None`:
```rust
pub(crate) fn try_gemm_with_phase_into(...) -> Result<()> {
    try_gemm_with_phase_and_scratch_into(ctx, weight, x, out, phase, None)
}
```

要让 prefill 用 scratch:
1. 创建 `MarlinPrefillScratch`(容量 m × k,m=4096 max → 10.5M INT8 = 10MB per scratch)
2. Prefill code path(`qwen3/prefill.rs`)接受 scratch 参数
3. 调用 `try_gemm_with_phase_and_scratch_into(...Some(&mut prefill_scratch))`
4. 改 `run_marlin_w4a8_linear_with_scratch` 让 m 可变(目前是 decode-fixed)

**Gain estimate**:
- B7 c=4 prefill 主成本是 GEMM compute(per nsys decomposition,prefill::compute = 97% active GPU time per `aaf0b55`)
- Alloc overhead 估计 < 5% of prefill total time
- → TTFT improvement ~2-5%

**LOC**:200-300 + scratch lifecycle + multi-config test
**Risk**:中(scratch sizing + lifetime + thread safety)

→ KILL at audit:LOC 大 + gain 小(prefill 不是 alloc bound)。

### Axis #1 已 KILLED(`f2bde6b`)

QQQ main thread_config dispatch port:Qwen3-4B 已命中 same default `(128, 128)` / `(64, 256)`,fallback configs 不会 fire。

### Axis #4 #5 未在本 brief 范围

- #4(graph capture hoist):已在 codex queue,等 codex pickup
- #5(sm_89 tile re-tune):需 ncu wrapper migration 先解锁

## 综合策略 — 切 axis

**结论**:W4A8 kernel + adapter 当前实现**已 near-optimal on Qwen3-4B sm_89**。
单维度改动 expected gain < 5%,不值得 +130-300 LOC commit。

**真正能 move needle 的方向**:

1. **等 codex #24 完成**(graph capture hoist for W4A8)— 已在 queue,无需 Claude 重复
2. **切到完全不同 axis**(per pickup queue 和 strategic ROI 评估):
   - **Spec decode**(M_spec/M_medusa)— per-token decode latency 改 50% (Medusa) 或 30% (external draft)
   - **Chunked prefill**(架构层) — prefill 性能突破点
   - **xgrammar FFI**(JSON / structured output)— different metric axis
3. **ncu 实测 W4A8 binding constraint**(when wrapper unblocks)— 量化数据决定值不值得 fp16→bf16 fuse OR sm_89 tile re-tune

**Claude 当前轴**:可以独立的(<100 LOC)work。最值得的:
- ✅ 写 Phase 0 audit briefs(DONE)
- ⚠ 等 codex 释放 GPU + 跟进 codex SGLang baseline 数据
- ⚠ pickup queue housekeeping

## ROI 总结

**Phase 0 audit 节省**:
- KILL Axis #1 救 +130-150 LOC port + ~30 cubin 编译时间 + bench failure
- KILL Axis #2 救 100-200 LOC kernel rewrite + accuracy verify(估计 1-2 days)
- KILL Axis #3 救 200-300 LOC adapter rewrite + multi-config test(估计 1 day)

**总 saving**:**~3-4 days codex / Claude work** prevented from low-ROI commit。

## Cross-references

- Phase 1 QQQ port KILL:`docs/experience/errors/2026-05-09-w4a8-qqq-port-phase1-killed.md`
- nsys 4-phase decomposition(prefill = 97% active):`docs/research/2026-05-09-eod113-p1a-nsys-decomposition-evidence.md`(`aaf0b55`)
- 业界 survey:`docs/research/2026-05-09-w4a8-industry-kernel-survey.md`
- B7 baseline:`docs/experience/wins/2026-05-09-baseline-snapshot-d4c3fc3.md`(TTFT 1614 ms,ITL 23.2 ms)
- pickup queue #24 in codex:`docs/plans/codex-pickup-queue-2026-05-09.md`
- skill v1.7.0 anti-pattern #18 Phase 0 substrate audit

## Rule

**不动手前 Phase 0 audit 全 candidate axis matrix,license-or-kill 各 axis
LOC vs expected gain。If 全 axis < 5% gain → 切完全不同 axis,不要硬 push
低 ROI 改动**。

W4A8 axis 已经过 sm_89 特化优化(L2 hint + small/large batch dispatch),
business kernel 改动饱和。下一步 needle-mover 不在 W4A8 internal,在
spec decode / chunked prefill / 或 codex #24 graph capture hoist 完成后
重新评估。

## 状态

W4A8 5 个候选 axis 全 audit。**3 个 KILL at audit**(Phase 1 / fp16-bf16 /
prefill scratch),**2 个 yield to others**(#24 codex queue / ncu blocked)。
Claude 当前 axis 转 strategic 评估 OR 等 codex GPU 释放后协同。
