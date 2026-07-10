# SchedulerMixedPolicy::Mixed A/B (#38) — KILLED at Phase 0 source audit

## Context

Per `docs/research/2026-05-09-prefill-axis-reopened-multi-key-bucket.md` §3
P1 candidate:`SchedulerMixedPolicy::Mixed` A/B(0 LOC,immediate bench)
对照 codex baseline 4k/c=4 1639.3 ms 看 mixed mode 是否给 ≥ +10% TTFT。
Phase 0 source-grep `infer/src/model/qwen3/forward.rs:664-674` 之前未做。

## Root Cause

`infer/src/model/qwen3/forward.rs:664-674`:

```rust
fn supports_mixed_batch(&self, kv_pool_format: KVFormat) -> bool {
    self.prefill_uses_paged_pool()
        && self.lora.is_none()
        && !self.uses_hybrid_w4_marlin()    // ← W4 hybrid 模式 NOT supported
        && matches!(
            kv_pool_format,
            KVFormat::BF16 | KVFormat::FP8E4M3 | KVFormat::INT8
        )
}
```

`infer/src/scheduler/cuda/execution.rs:419-425`:

```rust
} else if self.config.mixed_policy.allows_mixed()
    && self.model.supports_mixed_batch(self.paged_kv_pool.format)
{
    StepPlan::Mixed(candidates)
}
```

→ Mixed mode **架构性 不支持 W4-hybrid Marlin 模型**。Codex N=3 baseline
(`wins/2026-05-09-bench-sglang-reverify-post-p1.0-p1.2.md`)用
**`Qwen3-4B-W4-hybrid-zpfix`** = production W4-hybrid target → `--scheduler-mixed-policy mixed`
falls through to **Split**(execution.rs:439)。

## What Failed

之前 P1 ranking(`docs/research/2026-05-09-prefill-axis-reopened-multi-key-bucket.md`)
列 mixed-policy A/B 为 "0 LOC,可立即 bench" 而 **未做 model-architecture 兼容性
audit**。如果不 source-grep 直接跑 bench:
1. 启 ARLE server with `--scheduler-mixed-policy mixed --model-path infer/models/Qwen3-4B-W4-hybrid-zpfix`
2. Server **silently falls back to Split**(无 panic,无 warning,只是 step plan 不进 Mixed branch)
3. Bench 跑 60s × N=3 ≈ 1h wall-clock
4. 数据显示 Δ ≈ 0%(因为实际等同 baseline)
5. 报告"Mixed mode 无效",写 errors entry

→ **1 hour wasted** on bench that was architecturally guaranteed to be no-op。

## Fix(Phase 0 audit save)

Source-grep `supports_mixed_batch` **before** scoping P1 axis bench → 立即
发现 `!self.uses_hybrid_w4_marlin()` restriction → **KILL** #38 on
W4-hybrid baseline model。

## Mixed mode 在哪里仍可测(下放为 P3,不阻塞 P0)

| Model | Mixed 支持? | 是否 production target? |
|-------|------------|----------------------|
| `Qwen3-4B`(plain BF16)| ✓ | ❌(BF16 不是 production W4 主推)|
| `Qwen3-4B-AWQ` | ? need check `uses_hybrid_w4_marlin` 触发条件 | -- |
| `Qwen3-4B-GPTQ-W4A16-marlin-zpfix` | ? | -- |
| `Qwen3-4B-GPTQ-W4A8-marlin` | ? | -- |
| `Qwen3-4B-W4-hybrid-zpfix`(codex baseline)| ❌ KILL | ✓ |

Mixed-on-BF16 A/B 仍可作为 **separate P3 axis**(B2 baseline 2009 ms vs Mixed,
预估 < 5% gain because SGLang 默认 mixed=False)。但**不替代 W4-hybrid 上 P0
multi-key prefill graph**(#37)。

## Substrate-fix 候选(separate axis,需 codex)

如果想让 Mixed 在 W4-hybrid 上工作 → 需 codex 实施:
- 移除 `!self.uses_hybrid_w4_marlin()` restriction
- 实现 `forward_mixed_batch` for hybrid W4 path(BF16 prefill + W4A8 decode in 同 launch)
- 估计 200-500 LOC(decode + prefill kernel 协调)
- 风险:中(W4A8 + BF16 Marlin co-launch tile config 协调)

→ 暂不 pickup,等 #37 multi-key prefill graph(更高 ROI 直接 close 76.6% gap)
landed 后再评估。

## Rule

**Skill anti-pattern #18 Phase 0 substrate audit before scoping new wiring** —
"0 LOC CLI flag" 单变量 A/B 看似 cheap,但 **必须 source-grep 模型/路径
gating** 确认 flag 实际生效。**未 source-grep 就 license bench 是 hand-wave**。

每个 single-variable A/B brief 必须包含:
1. ✓ CLI flag 名 + parse 路径
2. ✓ 实际生效 gating(model trait / kv format / feature flag)
3. ✓ Fallback path 存在 → A/B 是否 silent fallback to baseline?

✗ 缺 #2 #3 → KILL bench at audit。

## Cross-references

- 死前 P1 brief:`docs/research/2026-05-09-prefill-axis-reopened-multi-key-bucket.md` §3 P1
- Source gating:`infer/src/model/qwen3/forward.rs:664-674`,`infer/src/scheduler/cuda/execution.rs:419-440`
- W4-hybrid baseline:`wins/2026-05-09-bench-sglang-reverify-post-p1.0-p1.2.md`
- skill anti-pattern #18:Phase 0 substrate audit

## 状态

#38 SchedulerMixedPolicy::Mixed A/B (P1) **KILLED at Phase 0 source audit**
on W4-hybrid baseline。Mixed-on-BF16 降为 P3(separate axis,non-production
target)。Real P0 unchanged:#37 multi-key prefill graph(直接 close 4k/c=4
+76.6% gap)。
