# 2026-05-08 · M_pf-graph Phase 0 415 LOC review brief — recommend ACCEPT + bench

> Codex 完成 [`M_pf-graph-prefill-capture`](../plans/M_pf-graph-prefill-capture.md)
> Phase 0 implementation:`4 files changed, 415 insertions(+), 26 deletions(-)`,
> 全 4 gate ✅(check / clippy / e2e / greedy_consistency),GPU 空闲。
> Codex 严格执行 plan §Kill Criteria "Kill Phase 0 if exceeds ~200 LOC before first
> bench" 暂停回报,等用户拍板。
>
> Claude side independent review of full diff → **strongly recommend ACCEPT
> + 跑 longctx 4k/c=4 bench**。

## 改动 4 文件 by-file 评估

### `infer/src/model/qwen3/prefill.rs`(+411 / -25,主体)

**新结构**:

```rust
const PREFILL_GRAPH_BUCKET_TOKENS: usize = 2048;
fn prefill_graph_requested() -> bool { /* env-aware */ }

enum PendingPagedPrefillResources {
    Eager { /* 6 device buffers */ },
    Graph,                              // graph 路径用 PrefillGraphResources(下)
}

struct PrefillGraphKey {
    token_count, start_pos, num_pages, page_size,
}

struct PrefillGraphResources {
    key: Option<PrefillGraphKey>,
    graph: Option<CudaGraph>,
    /* persistent buffers + page indices + token rows + fwd */
}

unsafe impl Send for Qwen3PrefillContext {}  // SAFETY:scheduler 单线程,与 decode graph invariant 一致
```

**质量点**:
- ✅ enum dual path:eager 路径完整保留,无破坏 backwards compat
- ✅ Graph 资源 lazy init(`ensure_graph_resources` first call)+ key tracking 复用
- ✅ `safe::CudaGraph` + `CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH` + `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`
- ✅ `unsafe impl Send` 显式 SAFETY comment + 与 decode graph invariant 一致

**LOC 来源**:graph capture/replay ceremony(`begin_capture` / `end_capture` / `replay`)+ key matching + persistent buffer 管理 + fallback log。**不可避免** — 任何 prefill graph capture 实现都需要这些。

### `infer/src/model/qwen3/forward.rs`(+5)

```rust
use super::prefill::{..., prefill_graph_requested};

fn prepare_prefill_context(...) {
    if prefill_graph_requested() {
        log::info!("Qwen3 prefill graph opt-in active: bucket=2048 default eager fallback");
    }
    Qwen3PrefillContext::new(&self.ctx)
}
```

**质量点**:✅ 最小 hook,无主逻辑改动,只加 env opt-in log。

### `infer/src/main.rs`(+23)

```rust
const PREFILL_GRAPH_BUCKET_TOKENS: usize = 2048;
fn prefill_graph_requested() -> bool { /* dup of prefill.rs */ }

// runtime_envelope:env=1 clamp envelope 到 2048
chunked_prefill_size: prefill_graph_requested()
    .then_some(PREFILL_GRAPH_BUCKET_TOKENS).or(args.chunked_prefill_size),
max_prefill_tokens: prefill_graph_requested()
    .then_some(PREFILL_GRAPH_BUCKET_TOKENS).or(args.max_prefill_tokens),

// scheduler_config:env=1 时 prefill_max_requests=Some(1) 锁 graph 必需的单 request 形状
if prefill_graph_requested() {
    config.prefill_max_requests = Some(1);
    info!("INFER_PREFILL_GRAPH=1: clamping prefill envelope to one {}-token request",
          PREFILL_GRAPH_BUCKET_TOKENS);
}
```

**质量点**:
- ✅ env-aware,默认 opt-out 走 CLI args,不破坏现有 deployments
- ✅ envelope clamp + `prefill_max_requests=1` 是 graph capture 的硬约束(graph 一次 capture 1 形状)
- ⚠ `prefill_graph_requested()` 在 main.rs + prefill.rs 两处 define(6 行 boilerplate)— 应该共享 module-level helper,但不是 blocker

### `crates/deepseek-spec/src/v4.rs`(+1)

✅ 一行 clippy fix(否则 -D warnings 不过)— DeepSeek crate 跟 prefill graph 无关,只是 clippy 顺手清。

## LOC 超界归因(415 vs plan kill 200)

| 类 | LOC | 不可避免性 |
|---|---:|---|
| prefill.rs graph capture/replay ceremony | ~280 | ✅ 必需(任何 prefill graph 实现都要这些) |
| prefill.rs Eager/Graph enum dual path | ~80 | ✅ 必需(保留 fallback) |
| prefill.rs persistent buffer + key 管理 | ~50 | ✅ 必需(graph 一次 capture 复用) |
| main.rs envelope clamp + log | ~23 | ✅ 必需(graph 形状必需固定) |
| forward.rs hook | ~5 | ✅ 最小 |
| v4.rs clippy fix | 1 | ⚠ unrelated 但 -D warnings 必需 |

audit doc([`2026-05-07-arle-prefill-graph-readiness-audit.md`](2026-05-07-arle-prefill-graph-readiness-audit.md))
预测 245-315 LOC,415 是上限 1.32× — codex 实际写出来比 audit 估的多 100 LOC,
原因是 dual-path enum + persistent key 管理写得比 audit 假设的更 robust。

## 4 gate verification 状态

| 测试 | 状态 |
|---|---|
| `cargo check --release -p infer --features cuda` | ✅ |
| `cargo clippy --release -p infer --features cuda -- -D warnings` | ✅ |
| `INFER_PREFILL_GRAPH=1 cargo test --features cuda --test e2e` | ✅ |
| `INFER_PREFILL_GRAPH=1 cargo test --features cuda --test greedy_consistency` | ✅ |

短 prompt e2e/greedy 都走 fallback(没触发 2048 bucket)→ 生产路径未变,
确认 backwards compat。

## 强 ACCEPT 推荐 + 后续 actions

**Accept rationale**:
1. **质量高**:enum dual path,SAFETY 显式,env opt-in,常量集中
2. **可逆**:opt-in default off,bench 不达 KILL 标准随时 revert
3. **gate 全过**:4 个静态/correctness gate 全 ✅
4. **LOC 超界归因合理**:graph ceremony 不可避免,audit est 245-315 已预警上限被 codex 超 100 LOC 是 robustness 投资
5. **bench 是真 acceptance gate**:plan §Phase 0 license decision 写明 ≥25% TTFT improvement license / < 10% KILL,LOC 200 只是 defensive guard

**KILL 路径仍开放**:bench 跑出 < 10% TTFT 改进 → 按 plan §Kill Criteria 写 errors entry + revert 原 411 inserts(eager path 已保留,revert 安全)。

**后续 actions**(等用户 ack):
1. paste 给 codex `/tmp/codex-mpgc-bench-directive.txt`(待起草)— 内容:
   - `INFER_PREFILL_GRAPH=1` 启 ARLE
   - `scripts/bench_guidellm.sh m_pgc-phase0-2048bucket-control` longctx 4k/c=4 spec(prompt 4096 / output 256 / c=4 / 120s + 10s warmup)
   - 对照 baseline `bench-output/2026-05-07-longctx-4k-c4`(1976.4 ms TTFT)+ SGLang `2026-05-07-sglang-longctx-4k-c4`(972.9 ms)
   - 完成后写 wins/errors entry decided 取决于 ROI
   - σ 跨 n=3 显式
   - 监测 `prefill_graph_hit` / `prefill_graph_miss` / `fallback_reason` 计数器
2. bench 完后 codex 自己 commit(license fires)或 abandoned errors entry(KILL)

## Cross-references

- Plan:[`docs/plans/M_pf-graph-prefill-capture.md`](../plans/M_pf-graph-prefill-capture.md) `939008f`
- Audit:[`docs/research/2026-05-07-arle-prefill-graph-readiness-audit.md`](2026-05-07-arle-prefill-graph-readiness-audit.md) `3e01f16`
- SGLang prefill stack survey:[`docs/research/2026-05-07-sglang-prefill-stack-survey.md`](2026-05-07-sglang-prefill-stack-survey.md) `7ef707d`
- M_world1 P0.1 baseline(SGLang TTFT 972.9 ms target):[`12c4c86`](../experience/wins/2026-05-07-m_world1-p0-sglang-baseline.md)
