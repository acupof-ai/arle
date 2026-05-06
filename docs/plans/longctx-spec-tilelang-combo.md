# Long-context × Spec-decode × TileLang AOT — World-best Combo

> Drafted 2026-05-07 by Claude (manager track). Companion to:
> - [`backend-unification.md`](backend-unification.md) — Codex's CUDA↔Metal收敛主线
> - [`dsv4-small-repro.md`](dsv4-small-repro.md) — Codex的从零预训 substrate

## 0. Why this plan — the combinatorial gap

ARLE 已有四条独立强项:
1. **TileLang AOT 内核**(`crates/cuda-kernels/tools/tilelang/`)— 比 Triton/CUTLASS 简短,per-SM 静态特化,可生成动态形状 cubin。
2. **Qwen3.5 hybrid attention**(`crates/qwen35-spec/`)— linear-attn(GDR) + full attn(sliding window) 混合,长上下文天然 O(L) 内存。
3. **Tier-KV cache**(`infer/src/kv_tier/` + `crates/kv-native-sys/`)— GPU/CPU/local-disk 三级,RadixCache 复用前缀。
4. **Continuous batching scheduler**(`infer/src/scheduler/cuda/`)— SGLang-equiv 调度面。

**问题**:四条都各自独立。**没有任何一家** vLLM / TGI / SGLang / TRT-LLM / mlx-lm 把这四条同时做了 — 行业里要么强 spec-decode + 短上下文(TRT-LLM),要么强长上下文 + 无 spec-decode(SGLang HiCache),要么强 tile-fusion + 单请求(TileLang 上游 demo)。

**世界第一的洞**:把四条 fuse 成一个 pipeline:
- 长上下文(≥64k)的 prefix 走 Tier-KV(T2 disk)+ Hybrid linear-attn 做 cheap re-embedding
- 解码用 EAGLE/Medusa 风格的 tree spec-decode,但是把 draft head 和 verify head **fuse 进同一个 TileLang AOT kernel**(单 launch,共享 shmem)
- Continuous batching 调度同时扛 prefill+decode+spec-verify,batched 对齐 by mixed-batch IR(M3 完成后)

这是一个 **真正的 four-way 组合**,有可发表性,也有 bench 收益。

---

## 1. 现状盘点 — 四条线接面

### 1.1 TileLang AOT — 当前覆盖

`crates/cuda-kernels/tools/tilelang/` 现有五个 Python kernel 模板:
- `batch_prefill_paged_hd128.py` / `batch_prefill_paged_hd256.py`
- `batch_decode_paged_hd256.py`
- `gated_delta_rule.py`(GDR prefill 三段:prepare/cumsum/chunk_a)
- `gen_tilelang_aot.py`(AOT 驱动)

**缺**:tree-spec verify kernel(把 K 个候选 draft token 一次性 score,共享 prefix)、draft generator kernel(轻量小模型 forward)、merged draft+verify kernel(本计划核心)。

### 1.2 Qwen3.5 hybrid

`crates/qwen35-spec/` + `infer/src/model/qwen35/` 已实现 GDR linear-attn 层 + 部分 full-attn 层交错。这就是天然的 long-context 友好结构。

**缺**:linear-attn 层在 spec-decode 下的等效 KV update(linear-attn 的 hidden state 是 recurrent,要支持 verify-rollback)。

### 1.3 Tier-KV

`infer/src/kv_tier/` T0(GPU)/T1(CPU)/T2(disk)三级,有 RadixCache。

**缺**:与 spec-decode 协同 — verify 失败回滚要丢弃 K 个虚拟 token 的 KV;长 context 的"非热"前缀按 LRU 下沉到 T2,但 spec verify 的 draft 命中往往落在 T1/T2,需要 lookup-aware spec policy。

### 1.4 Continuous batching scheduler

`infer/src/scheduler/cuda/` 已 SGLang-equiv。

**缺**:spec-decode batched verify(单 step 处理 K 个候选 token)的调度路径。

---

## 2. Milestones

每个 milestone 假设 backend-unification 已经到 M3(统一 schedule IR)。在那之前本计划只能做 §M_a 的 prep work(API + benchmark harness),不能做核心 fuse。

### M_a — Spec-decode bench + runtime knob(独立可做,与 unification 并行)

**Reality check (2026-05-07)**:`infer/src/speculative.rs` 已有 721 行框架,
`spec_decode_correctness` + `magicdec_self_spec_integration` 共 9 个测试
全部 ok(self-spec / external-draft / sparse self-spec / persistent state)。
真正缺的是**生产路径 + bench harness**:

1. **`arle serve` CLI 缺 `--num-speculative-tokens K` / `--spec-mode {self,external,sparse}`**
   开关。现在 spec-decode 框架只在测试里挂得上,CLI/HTTP 跑不进去。
2. **`scripts/bench_spec_decode.sh`**(新,wrap `bench_guidellm.sh`),先开
   `INFER_DETERMINISTIC=0`(production fast path)跑 vanilla baseline,再
   开 spec-decode 同 prompt set 跑一次,出 throughput / TTFT / acceptance-rate
   对比表。
3. **acceptance rate metric**:`SpecMetrics` 已存在但是没接到 `ServerMetrics::snapshot_engine_telemetry`(M1 的 EngineTelemetry);加一行让 acceptance rate 走 telemetry,bench 脚本可以 scrape。

**Acceptance**:
- `arle serve --num-speculative-tokens 4 --spec-mode self` 端到端跑通,Qwen3-4B 上 acceptance ≥ 0.6。
- `scripts/bench_spec_decode.sh self-spec-baseline` 产出 wins entry,vanilla vs spec-decode 矩阵 ≥ 1.4× decode-heavy throughput。
- `EngineTelemetry::spec_acceptance_rate` 字段 + `/v1/stats` JSON 渲染。

### M_b — TileLang fused draft+verify kernel

用 TileLang 写一个 kernel,把 draft step + verify scoring fuse 进单 launch。共享 prefix K/V 的 shmem load,共享 RoPE cache。Draft 用 K=4 树宽,verify 一次产出 4 个候选的 logits。

**Acceptance**:
- 比 M_a 的 separate-launch 版本快 ≥ 30%(理论上界 ~50%)。
- TileLang AOT 编译产物 ≤ 12 MB cubin 总量(per-SM × 几个 head config)。

### M_c — Hybrid + spec-decode 组合

linear-attn 层在 spec-verify 下需要的 hidden state checkpoint/rollback:
- Verify 前快照 GDR hidden state(O(num_layers × num_kv_heads × head_dim) ~ 几 MB)
- Verify 后只 commit 接受的 token 对应的 hidden state delta
- 不接受则 rollback 到快照

**Acceptance**:
- Qwen3.5-4B + spec-decode 在 32k context decode 上拿到 ≥ 1.5× speedup,准确性等价(greedy 字节级 match)。

### M_d — Tier-KV × spec-decode 协同

- Spec verify 时如果 prefix 命中 T1/T2,先 prefetch 到 T0,verify 时 K/V 已在 GPU。
- 失败 verify 不污染 RadixCache(rollback 不增加 cache entry)。

**Acceptance**:
- 64k context + spec-decode + Tier-KV(T0:T1 = 8GB:24GB)端到端 throughput vs SGLang HiCache + 无 spec ≥ 1.8×。

### M_e — 世界第一对照

把以上四个 fuse 起来,形成完整 pipeline:
- 32k/64k/128k context × 1, 4, 16 concurrency × Qwen3-4B / Qwen3.5-4B 矩阵
- 对手:SGLang(已知 1.609× 那个 baseline)、vLLM、TRT-LLM、mlx-lm

**Acceptance**:
- ≥ 6/9 workload 第一名(M6 of unification 已给框架,只是再扩 spec-decode 列)。

---

## 3. Risks

- **TileLang 0.1.9 版本能力**:fuse draft+verify kernel 依赖 TileLang 支持 multi-output 同步;若不支持降级到 separate-launch 但保留 prefix shmem 复用。
- **GDR rollback 状态空间**:每层 hidden_state ~ kv_dim × hidden,如果 num_layers ≥ 60 单次 snapshot 可能 ≥ 数百 MB。回退方案:只对 spec-verify 段(typically 前几层)做 snapshot,后面层用 deterministic re-replay。
- **spec-decode 与 batched-decode 数值漂移**:目前未修的 deferred bug([2026-04-13](../experience/errors/2026-04-13-batched-decode-high-concurrency.md))会让 spec-verify 看到的 logits 在 B=K vs B=1 不一致 → 接受率下降。M_a 落地前必须先 close 那个 bug 或确认其 ULP 漂移不影响接受率(需测量)。

---

## 4. Out of Scope

- 不做新的草稿模型架构(Medusa head / Hydra head 等)— 用现成的 EAGLE-1 即可。
- 不做 sequence-parallel 长 context training(M_a 只 inference-side)。
- Metal 路径不在本计划范围,等 backend-unification M5 把 Metal 接到 `ModelForward` 后再追加 Metal spec-decode entry(单独 plan)。

## 5. Definition of Done

- M_a 到 M_e 全部 acceptance 通过。
- `docs/experience/wins/<date>-spec-tilelang-hybrid-tier-combo-{cuda,metal}.md` 各一份对比表。
- 在 4070 Ti SUPER + (远端 H100) 上分别提交 final bench,对比矩阵 commit hash 可复现。

## 6. Sequencing & Ownership

| Track | Owner | Depends on | When |
|---|---|---|---|
| backend-unification M1-M3 | Codex 主线 | — | 现进行中 |
| 本计划 M_a | Claude(管理者) drafts brief, Codex implements | 不依赖 | unification M2 完成后启动(~Week 3) |
| M_b TileLang fused kernel | Codex(TileLang 熟手) | M_a + unification M3 | Week 4-5 |
| M_c Hybrid rollback | Codex,Claude review | M_b + unification M3 | Week 5-6 |
| M_d Tier-KV 协同 | Claude drafts API,Codex impl | M_c + unification M2 | Week 6-7 |
| M_e bench gauntlet | Claude(数据 + 报告) | 全部前置 | Week 7-8 |

并发可见:本计划与 unification 在 M3 之后开始大规模合流,M3 之前各自独立推进。

## 7. References

- `docs/plans/backend-unification.md`
- `docs/plans/dsv4-small-repro.md`
- `infer/src/speculative.rs`(现有 stub)
- `infer/tests/spec_decode_correctness.rs`(现有测试骨架)
- TileLang prefill/decode kernel 模板:`crates/cuda-kernels/tools/tilelang/`
- EAGLE-1 论文:Yuhui Li et al. NeurIPS 2024
