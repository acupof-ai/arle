# M_b — TileLang fused draft+verify spec-decode kernel

> 子计划 of [`longctx-spec-tilelang-combo.md`](longctx-spec-tilelang-combo.md) §M_b。
> 前置:M_a 已完成(`d58e274`,acceptance-rate 接 EngineTelemetry),
> backend-unification M3 不强依赖(M_b 在 kernel 层,不动 scheduler IR)。

## 0. Why fuse

现有 spec-decode 路径:每个 spec step 是 **2 次独立 launch** —— 先一次 draft model forward 产 K 个候选 token,再一次 target model verify(把 K+1 个位置的 K/V 都跑一遍 attention)。

两次 launch 之间有 launch overhead + L2 cache miss(prefix K/V 在 draft pass 里 load 过,verify pass 里再 load 一次)。L4 / 4070 Ti 上单步 spec decode 的 launch overhead ≈ 30-50 µs,一个步骤里两次 attention 也要 30-100 µs,**verify-launch 约占 spec step 时间的 30-40%**。

理论上界(测过):**fuse → ≥30% spec-step time 削减,折算到端到端 ≈ +10~15% throughput**(在 acceptance ≈ 0.7 的典型 self-spec 场景)。

业界没人做过 — vLLM / TGI / SGLang 都用 separate launch 的 EAGLE-1/Medusa/Hydra,只有 NVIDIA H100 spec-decode demo 用 PTX inline 实现过类似 fuse(non-portable, sm_90 only)。**TileLang 跨 SM 抽象正好是这件事的合适载体。**

## 1. 设计

### 1.1 入口 kernel signature

```python
def fused_draft_verify_paged_hd128(
    Q_target: T.Tensor((batch_size, num_q_heads * 128), bf16),
    # Existing paged KV
    K_pool: T.Tensor((num_pages, num_kv_heads, 16, 128), bf16),
    V_pool: T.Tensor((num_pages, num_kv_heads, 16, 128), bf16),
    KV_indices: T.Tensor((total_pages,), int32),
    KV_indptr: T.Tensor((batch_size + 1,), int32),
    KV_last_page_len: T.Tensor((batch_size,), int32),
    # Draft side
    Draft_W: T.Tensor((draft_dim, hidden_dim), bf16),  # tiny single-layer draft head, fits in shmem
    K_draft_buffer: T.Tensor((batch_size, K_max, num_q_heads * 128), bf16),  # next-K candidates' K
    V_draft_buffer: T.Tensor((batch_size, K_max, num_q_heads * 128), bf16),
    Q_draft_buffer: T.Tensor((batch_size, K_max, num_q_heads * 128), bf16),
    Verify_logits: T.Tensor((batch_size, K_max + 1, vocab_size), bf16),
    # Outputs
    Accept_count: T.Tensor((batch_size,), int32),     # 接受了多少 draft token (0..K)
    Bonus_token: T.Tensor((batch_size,), int32),       # bonus token 由 target sample 出
    # Shapes
    batch_size: T.int32,
    K_max: T.int32,        # max draft length per request
    sm_scale: T.float32,
)
```

### 1.2 Block schedule

Grid `(1, num_q_heads, batch_size)`,跟现有 HD128 prefill alias 一致。每 block 处理一个 (request × head)。

每个 block 的工作流(按时间序):

1. **Draft head pass**:从 shmem 中加载 `Draft_W`(假设 ≤ 100KB,4070 Ti SUPER shmem cap 100KB / SM,可以塞 0.6B 单层 draft 的一组 head)。target Q 的最后 hidden vec(注:这里需要 target 上一层 hidden 已经在显存可读,M_b 落地前先确认 caller 接口提供)。算 K 个 candidate token 的 logits → argmax → 产 K 个 token id。
2. **同 block 内 verify pass**:把 K+1 个 Q row(K candidates + verify pos)对 paged K/V 做 attention,同 block 内 reuse `q_tile`/`k_tile`/`v_tile` shmem buffer。每个 row 算 logits → softmax → top-1 → 跟 draft token 比对,greedy verify。
3. **rejection 处** break,bonus 在第一个 reject 位置 sample(greedy: argmax)。
4. 写 `Accept_count[bz]` + `Bonus_token[bz]`。

关键 trick:**prefix K/V 只 load 一次,跨 draft 和 verify 共享 shmem**。这是 fuse 的主要 win。

### 1.3 Shmem budget(L4 sm_89, 100KB / SM)

- Draft_W tile: 32KB(单层 draft head 的 W_q W_k W_v 子集)
- prefix K tile: 16 * 128 * 2 = 4KB / page,典型 8 page 加载窗口 = 32KB
- prefix V tile: 同样 32KB
- accumulator + softmax workspace: ~4KB
- **total: ~100KB**,刚好擦边 4070 Ti SUPER。H100 144KB 充裕。

### 1.4 Acceptance algorithm 选型

第一版 **only greedy verify**(temp=0)。matches `verify_tokens_greedy` in `infer/src/speculative.rs:97`。
温度采样 spec 留给 v2(需要 RNG state per-request)。

## 2. 任务拆分

| 任务 | 文件 | LOC 估 | Owner |
|---|---|---|---|
| P0. caller 接口确认(target hidden 提供给 fuse 入口) | `infer/src/scheduler/cuda/spec_path.rs` 调用 site survey | ~0,只 survey | Claude |
| P1. tilelang Python kernel 模板 | `crates/cuda-kernels/tools/tilelang/fused_draft_verify_paged_hd128.py`(新) | ~250 | Codex |
| P2. AOT generator hook | `crates/cuda-kernels/build.rs` 加 head config map | ~30 | Codex |
| P3. FFI bind | `crates/cuda-kernels/src/ffi/spec.rs`(新) | ~40 | Codex |
| P4. Rust dispatch | `infer/src/ops/spec_attention.rs`(新) | ~150 | Codex |
| P5. scheduler 接入(替换 separate-launch 路径) | `infer/src/scheduler/cuda/spec_path.rs` | ~80 | Codex |
| P6. correctness gate test | `infer/tests/spec_decode_correctness.rs`(扩) | ~50 | Claude |
| P7. bench(vanilla vs fused) | `bench_ab.sh` invocation 锁文档,跑 wins | ~50 | Claude |

总 LOC ~650,2-3 周。

## 3. Acceptance

- `cargo test --release -p infer --features cuda --test spec_decode_correctness` 全 pass(P6 后,新增 fused-path 等价测试)。
- vanilla self-spec vs fused self-spec 矩阵:fused 平均 ≥ +10% decode throughput,acceptance rate 不变(±1ppm)。
- TileLang AOT cubin 总量 ≤ 18MB(4 个 head config × 1 SM target)。
- `compute-sanitizer --tool memcheck` 0 errors 穿过整个 spec_decode 测试(已被 codex 过去用过的工具)。

## 4. Risk + retreat

- **风险 R1**: TileLang 0.1.9 不支持 `if .. else .. break` 早退(rejection 位置)。回退:在 K 个候选都 verify 完,后处理找第一个 reject 位置(轻微浪费,但不阻塞)。
- **风险 R2**: shmem 100KB cap 在 sm_89 下塞不下 draft_W + KV tiles。回退:draft_W 走 constant memory(64KB,read-only)。
- **风险 R3**: P1 完成但 perf 没 hit ≥10% target,可能是 launch overhead 不主导。回退:写 win doc 记录"fuse 不必要,separate-launch + L2 prewarm 同样 OK",revert P5 把 scheduler 切回 separate launch。

## 5. 不在范围

- Tree spec(K-branch tree drafting,vs flat K-token chain)— 留 v2,会让 draft pass 复杂化但 verify 部分完全不变,可独立追加。
- 温度采样 spec(non-greedy verify)— 留 v3,需要 RNG state plumbing。
- Hybrid (linear-attn + full-attn) layer 上的 spec-decode — 已是 M_c 子计划,需要 GDR rollback 能力。

## 6. References

- 父计划:[`longctx-spec-tilelang-combo.md`](longctx-spec-tilelang-combo.md)
- 现有 TileLang 模板:`crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py`(grid layout 范本)和 `batch_decode_paged_hd256.py`(decode-shape)
- 现有 spec-decode 验证逻辑:`infer/src/speculative.rs:90-180` (verify_tokens_greedy)
- EAGLE-1: Yuhui Li et al., NeurIPS 2024
- TileLang 0.1.9 release notes(检查 `if/else/break` 支持):`https://github.com/tile-ai/tilelang`
