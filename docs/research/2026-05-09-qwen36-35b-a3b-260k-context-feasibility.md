---
title: Qwen3.6 35B-A3B 4-bit @ 260k+ tokens — feasibility + phased plan
date: 2026-05-09
type: research
status: hypothesis-license
audience: ckl
---

# Qwen3.6 35B-A3B 4-bit 跑 260k+ tokens 可行性 + 分阶 plan

> 用户课题:把 `mlx-community/Qwen3.6-35B-A3B-4bit`(MoE,~19GB weights)在
> ARLE 上跑 **260k+ context tokens**。Phase 0 audit:hardware feasibility +
> ARLE feature gap + phased smoke→target plan + 主 risks。

## 1. Qwen3.6 35B-A3B architecture(per public spec)

| 维度 | 值 | 说明 |
|------|----:|------|
| Total params | 35 B | MoE total |
| Active params per token | 3 B(A3B)| 8 experts × top_k 路由,每 token 激活 ~3 B |
| 层数 | ~64(Qwen3-MoE 标准)| TBD verify with config.json |
| Attention heads | 32-64 | TBD |
| KV heads(GQA)| 4-8(GQA group_size 8-16)| TBD |
| Head dim | 128 | standard |
| Hidden size | 4096-5120 | TBD |
| Native context | 32768(常规)/ 长 ctx 训练版 65536+ | YARN/RoPE scaling 可扩 |
| 4-bit weight footprint | **~19 GB**(per CLAUDE.md `metal_serve` wired-limit 计算)| MLX 4bit pack |

**待确认 from config.json**(机器上无 cache,需要 Mac 端验证)。

## 2. KV cache memory 数学(关键 binding)

per-token KV cache size:
```
K + V = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element
```

假设 64 layers / 8 KV heads / 128 head_dim:

| KV format | bytes / token | 260k context | 注 |
|-----------|--------------:|-------------:|----|
| BF16 | 262 KB | **68 GB** | 不可能(超 M3 Ultra 192GB 一半)|
| FP8(E4M3)| 131 KB | **34 GB** | 可行,但需 ARLE Metal FP8 KV(待 confirm)|
| INT8 | 131 KB | **34 GB** | 同 FP8 size,质量稍差 |
| W4(2x packed)| 65 KB | **17 GB** | 极激进,需 W4 KV 路径(ARLE 未实现)|

**Total memory budget @ 260k**:
- Weights:**19 GB**(4-bit MLX)
- KV cache(FP8):**34 GB**
- 中间 activations / scratch:**~5-10 GB**(MoE expert routing + attention temp)
- **Subtotal:58-63 GB**

→ **必须 M3 Max 64GB(ceiling tight)OR M3 Ultra 192GB**(舒适 + 多并发)。M2/M3 Pro 36GB 不够。

如果走 W4 KV(预估 实现需 +500 LOC)→ subtotal 41-46 GB,M3 Pro 36 GB 仍紧 / M3 Max 64 GB 舒适。

## 3. Native context window

Qwen3.6 35B-A3B 原 train 上下文 32k token。**260k = ~8× 扩展**,需 RoPE scaling:
- **YARN**(已被 Qwen 系列 validate)— 可扩 4-8×,质量保持
- **RoPE NTK-aware**(基础)— 可扩 2-4×
- **PI(Position Interpolation)**— 可扩 4×

**推荐**:YARN with `factor=8`,native 32k → 260k ≈ 8× extend。**质量 risk 中**(Qwen3.6 没有 publicly validate 260k YARN 性能,可能 perplexity 退化 或 long-range retrieval task 失败)。

## 4. ARLE 当前 Qwen3.6 + 长 context 支持(per docs/support-matrix.md)

| 维度 | CUDA | Metal |
|------|------|-------|
| Qwen3.6 35B-A3B 4-bit load + run | ❌ stub `GPU required` | ✅ Beta(M4 Pro 2026-04-27 confirmed)|
| MoE substrate | ❌ | ✅(qwen35.rs 含 MoE)|
| 长 context bench | partial(L4 16384)| 仅 short diagnostic |
| KV format options | BF16 / FP8 / INT8 | BF16 default,FP8/INT8 支持 TBD |
| RoPE scaling(YARN/PI)| TBD | TBD |
| DFlash spec decode | -- | ✅ wired(`z-lab/Qwen3.6-35B-A3B-DFlash` draft)|
| Continuous batching | ✅ paged | ✅ `BatchKVCache` packed varlen |

→ **必须用 Metal backend**。ARLE Metal Qwen3.6 是 Beta,已能 short request smoke,但 **260k 长 context 未测**。

## 5. Feature gap audit(blockers to 260k)

| Feature | 当前状态 | 260k 需 | LOC 估计 |
|---------|---------|--------|----------|
| RoPE YARN scaling | TBD(可能未 wire)| ✓ | 50-150 if 未 wire |
| FP8 KV cache Metal | TBD verify | ✓(否则 BF16 KV 60+ GB)| 100-300 if 未实现 |
| Paged prefill chunked > 64k | 当前 chunk 2048 max | 可能需 > 8k chunk 减少 step 数 | 50-100 |
| Sliding window OR streaming attention | unlikely 已实现 | 可选(若 vanilla attention 不 fit)| 200-500 |
| Long-context attention 算法(StreamingLLM/H2O)| 未实现 | 可选(vanilla GQA 应能撑 if memory enough)| 500+ |
| Quadratic attention memory(prefill 阶段)| O(N²) attention temp | 260k² × heads × ~1MB temp = 巨大 | 用 FlashAttention/Online softmax(已有 `mlx-lm` BatchKVCache pattern)|
| MoE expert router efficiency | 已 wire | 验证 260k step 时 routing 不爆 | 0(但需 bench)|

## 6. 分阶 phased plan(Mac 端执行)

### Phase A — Smoke & feasibility(1-2 天)

License gate:Qwen3.6 35B-A3B 在 Mac 上能 inference 32k context with FP8 KV,
质量 PPL ≤ 1.05× baseline。

```bash
# 假设在 M3 Max 64GB Mac 上
cargo build --release --no-default-features --features metal

./target/release/metal_serve \
  --model-path mlx-community/Qwen3.6-35B-A3B-4bit \
  --port 8765 --max-running-requests 1 \
  --max-seq-len 32768 \
  --kv-cache-dtype fp8  # if 支持

# 32k smoke prompt 测 PPL + tok/s
curl -X POST http://localhost:8765/v1/completions ...
```

KILL signals:
- Server 不能 load(配置不兼容)
- 32k context PPL > 1.5× baseline(YARN 没 wire)
- 32k inference OOM 在 M3 Max 64GB

### Phase B — 64k extend + YARN scaling(2-3 天)

License gate:64k context PPL ≤ 1.10× baseline,tok/s ≥ 5(可用 floor)。

```bash
# YARN factor=2 拉到 64k
INFER_ROPE_SCALING='{"type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
./target/release/metal_serve --max-seq-len 65536 ...
```

需要在 ARLE Metal Qwen3.6 forward path 加 RoPE scaling 注入(if not present)。

KILL:64k OOM OR PPL > 1.5× baseline。

### Phase C — 128k(3-5 天)

License gate:128k context PPL ≤ 1.20× baseline,tok/s ≥ 3。

YARN factor=4。需 verify FP8 KV memory 充足:128k × FP8 = 17 GB。

KILL:128k OOM(需 W4 KV)OR PPL > 2× baseline。

### Phase D — 260k+(5-10 天 if Phase A-C smooth)

License gate:260k context **完成 inference 不 OOM**,first-token latency < 60s,
PPL ≤ 1.50× baseline。

YARN factor=8。Memory budget critical:
- Weights 19 + KV(FP8)34 + scratch 8 = **61 GB**(M3 Max 64GB ceiling tight)
- 如果 OOM,需 W4 KV(+200-500 LOC implement)

**Stretch**:bench 260k tok/s + 多并发 c=2 / c=4(需 ≥ 192 GB Ultra 才能多并发)。

## 7. 主 risks

1. **Hardware**:用户当前是否有 M3 Max 64GB+?CUDA box 不可用(Qwen3.6 stub)。
2. **Native context limit**:Qwen3.6 train 上下文未公开 confirmed 至 260k 的 YARN scaling 质量
3. **MLX MoE forward at 260k**:`mlx_async_eval` per-step encoding cost 可能成 binding(per CLAUDE.md `MLX_MAX_OPS_PER_BUFFER` 警告 in Qwen3.6 wash-or-loss)
4. **Quadratic attention temp**(prefill):260k × 260k attention matrix(per layer 头)— if not chunked + online softmax,可能 OOM 单 prefill step
5. **W4 KV cache Metal 路径未实现**:if FP8 不 fit,backend feature 缺失阻塞 260k

## 8. 推荐 next step

**待用户确认**:
1. **Hardware**:M3 Max 64GB / M3 Ultra 192GB / 其他?
2. **质量要求**:PPL 1.05× / 1.20× / 1.50× 哪个 acceptable?
3. **Throughput vs latency 优先**:260k single request 可,还是要 c=4 多并发?

**确认后立即可启 Phase A smoke**(若 Mac 端 ARLE 已 build)。

如果 hardware 仅 16GB CUDA(本机)→ **此课题不可行 on this box**,Qwen3.6 MoE 是 CUDA stub。

## 9. Cross-references

- ARLE Qwen3.6 support state:`docs/support-matrix.md:62`,`docs/environment.md:143`
- Metal canonical model 文档:CLAUDE.md "Metal canonical model"
- 32k-128k 项目(Qwen3-4B,non-Qwen3.6):`docs/projects/2026-04-30-longctx-32k-128k-leadership.md`
- DFlash Qwen3.6:`infer/src/backend/metal/dflash.rs`
- Metal Qwen3.6 short check 历史:`docs/experience/errors/2026-04-27-dflash-long-sequence-only.md`

## 10. 状态

Qwen3.6 35B-A3B @ 260k feasibility brief。**Hardware = M3 Max 64GB(tight)/ M3
Ultra 192GB(舒适)**。**最大 risk = ARLE Metal long-context 未 validate +
RoPE scaling 可能未 wire**。**Phased plan A→D**(smoke 32k → 64k → 128k → 260k)。
**等用户 confirm hardware**,然后 Phase A 立即可启。
