---
title: Chunked prefill 已存在 — strategic axis ranking 纠错
date: 2026-05-09
type: research
status: correction
supersedes: docs/research/2026-05-09-post-w4a8-strategic-axis-ranking.md §🥇 P0
---

# Chunked prefill axis 纠错 — substrate 已存在,不是 P0 新 work

> 接续 `2026-05-09-post-w4a8-strategic-axis-ranking.md`(列 P0 = chunked
> prefill scaffold,300-500 LOC,1-2w)。**纠错**:Phase 0 source audit
> 显示 ARLE chunked prefill substrate **已完整存在**,P0 ranking 此项作废。

## Phase 0 source-level evidence

`infer/src/scheduler/types.rs:265-273`:

```rust
pub fn pick_chunked_prefill_size_for_hbm(gpu_total_bytes: usize) -> usize {
    const GIB: usize = 1024 * 1024 * 1024;
    match gpu_total_bytes {
        n if n < 35 * GIB => 2048,   // 4070 Ti SUPER 16GB → 2048
        n if n < 60 * GIB => 4096,
        n if n < 90 * GIB => 8192,
        _ => 16384,
    }
}
```

`infer/src/backend/cuda/bootstrap.rs:411-433` — 已 print scheduling envelope
对照 SGLang-equiv 值,源代码 comment 直接 reference SGLang
`server_args.py`:

```rust
// SGLang reference values are sourced from
// `python/sglang/srt/server_args.py` (chunked_prefill_size HBM
// table, max_num_batched_tokens=16384, mem_fraction_static=0.85, …)
let sglang_chunk = match gpu_total_bytes / GiB {
    0..=34 => 2048, 35..=59 => 4096, 60..=89 => 8192, _ => 16384,
};
info!("Scheduling envelope (resolved | SGLang-equiv): \
       max_num_batched_tokens={} | 16384, \
       chunked_prefill_size={} | {}, …", …);
```

`infer/src/scheduler/cuda/prefill.rs` 实际运行时 log 印:

```
Request {}: chunked prefill starting (4097 effective tokens, chunk_size=2048)
```

→ ARLE chunked prefill **已对齐 SGLang HBM tier table**。c=4 longctx-4096
请求当前实际拆 2 chunks × 2048 tokens。

## ARLE 已实现 vs SGLang 完整对比

| 维度 | ARLE 当前 | SGLang | 状态 |
|------|----------|--------|------|
| chunked_prefill_size auto-pick(16GB)| 2048 ✓ | 2048 ✓ | **align** |
| max_num_batched_tokens | 16384 ✓ | 16384 ✓ | **align** |
| max_prefill_tokens | 16384 ✓ | 16384 ✓ | **align** |
| mem_fraction_static | 0.85 ✓ | 0.85 ✓ | **align** |
| mixed batch policy(decode + prefill 同 step)| `Split`(default)| `enable_mixed_chunk=False`(default) | **align**(都默认 split)|
| Mixed policy substrate | ✓(`SchedulerMixedPolicy::Mixed`)| ✓ | **align**(已有 substrate,opt-in) |
| Long prefill threshold | 4096 | -- | -- |

## 纠正:strategic axis ranking 重新评估

### 🚫 被作废 — chunked prefill scaffold(P0 from prior brief)

**原因**:substrate 已存在并对齐 SGLang。无需 300-500 LOC scaffold。

### ✅ 可保留 — chunked prefill **tuning** A/B(可选,需 GPU)

仍可作 single-variable A/B 候选:
- `chunked_prefill_size` 1024 / 2048(default)/ 4096 sweep on B7 c=4
- 评估非 align value 是否反而 perf 更好(SGLang default 不一定 sm_89 4070 Ti SUPER 最优)
- LOC = 0(纯 CLI flag),1 hour wall-clock × 3 runs

预估 gain:< 5%(per Phase 0 prediction:SGLang 早已经 sweep 过类似 GPU class)

### 🟡 真正 unfilled axis — `SchedulerMixedPolicy::Mixed` A/B

**Hypothesis**:c=4 mixed workload(prefill 4096-in + 已 admitted decode)开 mixed 模式
应该让 decode 不 stall 等 prefill chunk 间隙。
- LOC = 0(已有 substrate,只是 CLI flag `--scheduler-mixed-policy mixed`)
- 风险:低
- 需 GPU bench(c=4 longctx + c=8 multi-tenant)
- 预估 gain:hypothesis,需实测;mixed 在 SGLang 默认 false 暗示 perf 不一定显著

### 🥇 真正 P0 候选(待 codex SGLang baseline 完成 → gap quantify)

待 codex 47m+ N=3 SGLang baseline 完成,ARLE vs SGLang per-workload gap 出来后:

| Workload | ARLE 当前 TTFT | SGLang 预估 TTFT | gap |
|----------|----:|----:|----:|
| B5 W4A16 c=4 longctx | 2340 ms | ~928 ms(per codex 实测 partial)| **~2.5×** |
| B7 W4A8 c=4 longctx | 1614 ms | ~928 ms(用 awqmarlin 基线)| **~1.7×** |
| B5 W4A16 c=1 | 572 ms | ? | TBD |
| B7 W4A8 c=1 | 410 ms | ? | TBD |

**注意**:ARLE B5(W4A16 BF16-act)gap **更大** 比 B7(W4A8 INT8-act)。SGLang
awqmarlin = W4A16 BF16-act,**和 ARLE B5 同 quant scheme**,所以 B5 vs awqmarlin
是 apples-to-apples → 这是真正 unfilled 的 W4A16 kernel gap。

→ 真正 P0 候选(待 baseline confirm 后):
- **W4A16 Marlin kernel upgrade**(参考 SGLang awqmarlin 实现)
- **#30 hybrid W4A16/W4A8 dispatch**(prefill W4A16 + decode W4A8)
- **mixed batch policy A/B**(low-cost,先做)

## 教训(skill v1.7.0 anti-pattern #18 Phase 0 substrate audit)

**写 P0 scaffold brief 之前必须 source-grep 已有 substrate**。
- 之前 ranking brief 凭直觉假设 ARLE 没 chunked prefill → 写 300-500 LOC scaffold P0
- Source-grep 1 分钟就能发现 `pick_chunked_prefill_size_for_hbm` + `SchedulerMixedPolicy`
- **节省**:300-500 LOC codex work + 1-2w wall-clock,完全不 commit 错的 scaffold

**Rule**:nsys 实测 binding constraint(prefill 97% active GPU)≠ kernel
or scheduler substrate 缺失。这两轴正交。Substrate 缺失需 source audit
确认,不是 nsys 推断。

## Cross-references

- 错的 brief:`docs/research/2026-05-09-post-w4a8-strategic-axis-ranking.md` §🥇 P0(chunked prefill scaffold)
- ARLE chunked prefill substrate:`infer/src/scheduler/cuda/prefill.rs`,`infer/src/scheduler/types.rs:265`
- SGLang reference:`bootstrap.rs:411` comment 引 `python/sglang/srt/server_args.py`
- Mixed policy:`infer/src/scheduler/types.rs:510` `SchedulerMixedPolicy`
- B5/B7 baseline:`docs/experience/wins/2026-05-09-baseline-snapshot-d4c3fc3.md`

## 状态

Strategic axis ranking 中 P0 = chunked prefill scaffold **作废**。
真正 P0 候选待 codex SGLang baseline N=3 完成 → ARLE B5/B7 vs awqmarlin
gap 数据 confirm 后重新评估。当前最高 ROI 候选:**mixed batch policy A/B
(LOC=0 CLI flag)**,需 GPU。
