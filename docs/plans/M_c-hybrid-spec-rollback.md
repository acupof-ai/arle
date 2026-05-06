# M_c — Qwen3.5 Hybrid: spec-decode rollback for GDR linear-attn layers

> Sub-plan of [`longctx-spec-tilelang-combo.md`](longctx-spec-tilelang-combo.md) §M_c.
> 前置:M_b.2(sparse-self-spec fusion)已 land 或在 land,M3 unified IR 已 land。
> Goal:让 Qwen3.5 hybrid(linear-attn + full-attn 交错)模型在 spec-decode 失败 verify 时
> 把 linear-attn 层的 recurrent hidden state **正确回滚**,匹配 K-step prefix。

## P0 finding (2026-05-07) — snapshot infra already exists

Survey of `infer/src/model/qwen35/recurrent_state.rs` 暴露:

- `LayerRecurrentState { state: CudaSlice<f32>, conv_state: DeviceVec }` — 单层 GDR 状态。
- `RecurrentState::save_snapshot(ctx)` / `restore_snapshot(ctx, seq_len)` — 已实现,
  目前只为 prefix cache full-hit 用。**~49 MB / GPU memcpy for Qwen3.5-4B
  (24 layers × ~2 MB each)** 的成本注释在 `recurrent_state.rs:97`。
- `RecurrentState::seq_len` — 跟踪当前 state 对应的 token 数。

**这意味着 M_c 不是从零做 rollback 基础设施;它是把 prefix-cache 用过的 snapshot
机制重新连到 spec-verify 路径。** 整体工作量比原计划估的 1 周缩到 ~3 天。

## 1. Spec-rollback protocol

### 1.1 Step-by-step

设当前 spec step 提议 K 个 draft tokens,target verify 接受 k_acc ∈ [0, K] 个。
linear-attn 层(GDR)在 verify 之前必须知道:
- **保留**:当前 state(对应 prefill + 已 commit 的所有 token)
- **临时推进 K 步**:把 K 个 draft token 喂进 GDR 算 verify 的 hidden
- **rollback**:state 回退到 commit 后只前进 k_acc 步的位置

直接从 `seq_len` 视角看,这是 K-step transient + (K - k_acc)-step rollback。

### 1.2 Snapshot/restore choreography

```
[before spec step]
  state_old = current state (corresponds to seq_len = N committed tokens)
  
[spec verify forward]
  S1. snapshot_old = save_snapshot()         # ~49 MB memcpy
  S2. run K draft tokens through GDR forward # state advances K steps
  S3. record verify logits / K+1 logits
  
[verifier compares]
  if k_acc == K (full accept):
      no rollback needed; state already at N+K
      # bonus token from target gives K+1th, advance by 1 more (next spec step starts here)
  elif k_acc < K (partial / full reject):
      S4. restore_snapshot(snapshot_old)         # state ← N
      S5. for tok in accepted_drafts[..k_acc] + [bonus_target]:
              gdr_step(state, tok)               # state ← N + k_acc + 1
      # rollback complete
  
[discard snapshot]
  drop(snapshot_old)
```

S1 + S4 是 D2D memcpy,~49 MB × 2 ≈ 100 MB / spec step。Qwen3.5-4B
在 4070 Ti SUPER 上 L2 / HBM 带宽 ~700 GB/s,memcpy 耗时 ≈ 0.14 ms / spec step。
对比 spec step 整体 latency ~10-15 ms,**额外开销 < 1.5%**,acceptable。

### 1.3 Per-request scope

每个 active request 有独立 `RecurrentState`(已经是 `Qwen35State::recurrent_state`
field)。snapshot 也得 per-request 独立(目前的 `RecurrentSnapshot` field 在
`RecurrentState` 上,刚好是 per-request)。

**问题**:同时多 request batch 时,要不要把 snapshot 全部一次 D2D?
**答**:GDR 每层 state 在 GPU 上,batch 多 request 各自有 state,snapshot copy 也
是 per-request 的小 buffer。可以并行 issue,但 stream 顺序由 caller 保证。

## 2. 实现拆分

| 任务 | 文件 | LOC 估 | Owner | 注 |
|---|---|---|---|---|
| Q1. 把现有 `RecurrentState::save_snapshot/restore_snapshot` 提到一组 spec-friendly API:`spec_snapshot()` 返回 `RecurrentSnapshot` owned 句柄(不存 in-place);`spec_restore_into(snapshot)` 接受句柄 | `infer/src/model/qwen35/recurrent_state.rs` | ~80 | Codex |
| Q2. Qwen3.5 forward 加 verify hook:`forward_spec_verify(input_tokens=[committed]+drafts, …) → SpecVerifyOutput`,内部 S1-S3 序列。和 Qwen3 forward 的 SpecVerify 接口对齐 | `infer/src/model/qwen35/forward.rs` | ~120 | Codex |
| Q3. scheduler `spec_path.rs` 的 hybrid 分支:Qwen3.5 模型走 hybrid 路径时,verify 后把 accept count 反馈给 model,model 决定 rollback 序列(S4-S5)| `infer/src/scheduler/cuda/spec_path.rs` + `infer/src/model/qwen35/forward.rs` | ~70 | Codex |
| Q4. correctness gate:`spec_decode_correctness` 加 `spec_decode_qwen35_hybrid_greedy_is_bit_identical`,prompt 集 5 个,verify B=1 / B=3 都通过 | `infer/tests/spec_decode_correctness.rs` | ~70 | Claude |
| Q5. 性能微测:vanilla qwen35 vs spec qwen35,decode-heavy。要求至少不比 vanilla 慢 5%(不强求加速,因为 hybrid 的 full-attn 部分本来已经 spec 了,linear-attn 的 spec 收益小)| `bench_ab.sh` invocation 锁 | ~30 | Claude |
| Q6. wins entry | `docs/experience/wins/<date>-m_c-hybrid-spec-rollback.md` | 一个文件 | Claude |

总 LOC ~370,3 天工期。

## 3. Acceptance

- `spec_decode_correctness` 全 pass + 新加 hybrid 测试 pass。
- `cargo test --release --features cuda --test e2e_qwen35` 不回退。
- Memcheck 0 errors over `spec_decode_qwen35_hybrid_greedy_is_bit_identical`。
- Decode-heavy bench:vanilla qwen35 vs spec qwen35,后者不慢于前者 5%(snapshot D2D 开销 < 1.5% 理论值,留 3.5% 实测裕量)。
- wins entry 落地。

## 4. Risks + retreat

- **R1 — 多 request 并行 snapshot 拥塞 D2D 带宽**。10 个并发请求每 spec step ~1 GB
  额外 memcpy。HBM 700 GB/s → 还有 1.5% 时间,但跟其他 D2D(KV write、
  Q/K/V GEMM 输出)挤一起 latency 可能恶化。Retreat:per-request batched
  snapshot(把 N 个 request 的 state copy 合成一次 大 launch),减少 launch
  overhead;perf benchmark 覆盖 c=4 / c=16。
- **R2 — GDR rollback 数学错误**:GDR 的 `state` 是 ∑_t γ_t · K_t V_t^T
  形式,任何顺序错都会污染。Mitigation:Q4 加 byte-exact 字节对比测试,
  spec 路径输出和 vanilla 路径输出对应每 token 必须一致(`INFER_DETERMINISTIC=1`
  下)。
- **R3 — `forward_spec_verify` 跟现有 `forward_prefill` 路径分叉,引入两个 GDR
  forward 实现**:违反 CLAUDE.md 的"no half-states"。Mitigation:Q2 提
  GDR forward 公共部分到 helper `gdr_forward_step(state, tokens, snapshot=None)`,
  verify 是它的一个 mode。

## 5. 不在范围

- **DSv4-style hybrid(MLA + linear-attn 混合)**:DSv4 走单独 plan,本计划只覆盖 Qwen3.5。
- **Metal 端 hybrid spec**:M5 of unification 完成后才能动 Metal `forward.rs::run_metal_scheduler_runtime` 内部。本计划的 contract 设计兼容 Metal,但实现留 v2。
- **Spec verify 长 K**(K > 8):snapshot D2D 成本随 K 不变,但每个 spec step 接受率会下降到无收益区间。本计划假设 K ∈ [3, 6]。

## 6. References

- 父计划:[`longctx-spec-tilelang-combo.md`](longctx-spec-tilelang-combo.md) §M_c
- 现有 snapshot infra:`infer/src/model/qwen35/recurrent_state.rs:91-110` (`save_snapshot`/`restore_snapshot`)
- Qwen3 spec verify ref impl:`infer/src/model/qwen3/forward.rs:684`
- spec_path.rs current hybrid stub:`infer/src/scheduler/cuda/spec_path.rs:280-291`
