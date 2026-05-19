---
title: Metal — Qwen3.6 / Qwen3.5-MoE 原生 MTP speculative decode 设计
date: 2026-05-19
type: design
status: on-hold — draft 已落地,实施 deferred(2026-05-19)
owner: tbd
hold-reason: |
  P0 需要先下 71.9 GB BF16 模型 + 写自家 quant 脚本,投入大。
  现有 ARLE Metal serve 已 = MLX(86 tok/s),无 user-blocking 瓶颈。
  恢复条件之一:M4 Pro 带宽风险(§9 R1)被独立验证可破,或 ARLE 用户实际 hit 86 tok/s 上限报怨。
related:
  - docs/plans/2026-05-01-longctx-spec-decode-phase2.md
  - docs/plans/2026-05-10-dsv4-qwen36-substrate-audit.md
  - docs/support-matrix.md
  - crates/mlx-sys/src/mlx_dflash_draft_model.cpp
---

# Metal — Qwen3.6 / Qwen3.5-MoE 原生 MTP speculative decode 设计

> 一句话:Qwen 官方为 Qwen3.6 family 训练并 ship 了 MTP 头(`mtp.fc` + 单层
> MoE 子模块);用它当 self-drafter 替代 DFlash 的外部 draft model,**预期
> 在 M4 Pro 上把 Qwen3.6-35B-A3B 4-bit MLX 从 86 → 105–130 tok/s**,即首次
> 突破内存带宽天花板。

## 0. Status snapshot

| 项 | 当前 | 目标(本 design) |
|---|---|---|
| Qwen3.6 on Metal | Beta — load + serve via MLX | + 原生 MTP draft path |
| DFlash on Qwen3.6 | 不适用(文档自述"长上下文才决策") | 保留 dense 路径,MoE 走 MTP |
| MLX-parity perf | 85.6 tok/s decode = mlx-lm 86.3(ad-hoc bench 2026-05-18)| **≥ 105 tok/s decode** at temp=0 / `91-tok prompt / 256 out` |
| Weight source | `mlx-community/Qwen3.6-35B-A3B-4bit`(MTP stripped) | 自家 quant 脚本保留 `mtp.*`,或拉 `Qwen/Qwen3.6-35B-A3B` BF16 → ARLE MLX-4bit-MTP |

## 1. Motivation — 为什么是 MTP,为什么是现在

### 1.1 实测证据(2026-05-18 / 2026-05-19,M4 Pro 48GB,Qwen3.6-35B-A3B 4-bit)

| Runtime / 路径 | decode tok/s | 备注 |
|---|---:|---|
| MLX `mlx_lm.generate` 直跑 | 86.3 | 91/256, temp 0 |
| ARLE Metal serve(无 MTP)| 85.6 | OpenAI-compatible /v1/completions |
| llama.cpp baseline(无 MTP)| 38.4 | GGUF Q4_K_XL,`-fa on -np 1` |
| llama.cpp MTP `n_max=2` | 44.3 | **+15%** over its baseline,sweet spot |
| llama.cpp MTP `n_max=6`(Unsloth 默认)| 29.3 | **-24%** — defaults are wrong for MoE on M4 Pro |

**关键含义:**
1. ARLE Metal serve 已 = MLX(85.6 vs 86.3),**两个 runtime 都吃满了 M4 Pro
   273 GB/s 统一内存带宽的 ~78%**;纯解码不可能再涨。
2. llama.cpp MTP 即便最优 `n_max=2`,仍只是 MLX 的 52%,瓶颈在 GGUF runtime
   而非 MTP 算法。**ARLE 拿到 MLX-tier 的 runtime + MTP 算法,才是突破口。**
3. MTPLX(MLX-Swift 第三方,Qwen3.6-27B dense)实测 **2.24x decode @ temp 0.6**;
   MoE 的 acceptance 一般低一档,**保守估计 1.3–1.5x → 86 → 110–130 tok/s**。

### 1.2 为什么不是外部 draft model(DFlash 现有方案)

- DFlash 5.9× decode 是在 Qwen3-4B BF16 + 小 draft model 上,**MoE 一直未验**。
- 外部 drafter 在 MoE 上 acceptance 低(社区一致经验 < 50%),vLLM 实测 MTP 上
  到 70-85%(因为 MTP 头跟主模型同分布同 vocab,acceptance 接近完美)。
- DFlash 的 draft 是独立 transformer(`mlx_dflash_draft_model.cpp` 里 ~280B 7-proj
  layer 堆叠),**冷启动还要单独装载 draft model 权重**;MTP 头复用主模型 hidden state,
  零额外 RAM(除了 ~1GB 的头本身),零冷启动。

## 2. Scope

### 2.1 In

- Qwen3.6-35B-A3B 4-bit MLX,在 Metal serve(`infer` + `metal_serve` binary)上启用原生 MTP
- MTP 头的安全装载(从带 `mtp.*` 的 safetensors;包含从官方 BF16 转 4-bit 时保留
  这些权重的 quant 脚本)
- Draft → batched verify → Leviathan-Chen accept + residual correction 全 loop
- Temp=0 greedy + temp>0 sampling 两条 path(MTPLX 启示:T>0 才有 2x+)
- Bench harness:`scripts/bench_36_mtp.py` + 三档对照(no-spec / DFlash / MTP)
- CLI / env opt-in:`--spec-type mtp` 或 `INFER_SPEC_TYPE=mtp`,默认关

### 2.2 Out(本 design 不覆盖)

- CUDA 实现 — Qwen3.6 在 CUDA 仍是 stub(per support-matrix),由 #2 next-model
  优先级承接,本 design 只标接口契约不写实现
- Qwen3.5-MoE 之外的 family(Qwen3 dense / Qwen3.5 dense / DSv4)— DFlash 继续用
- 多 batch / concurrent decode 下的 MTP — Phase 1 只覆盖 single-stream;multi-tenant
  延后到 Phase 3+
- 长上下文(>8k)的 MTP behavior — DFlash 在长 ctx 下另有故事,正交问题
- MTP 头的训练/微调(only inference path)

### 2.3 显式不做的事(避免误解)

- **不替换 DFlash** — DFlash 在 dense 模型上 5.9x 是真实收益,留着。MTP 只是 MoE
  family 的并列选项。
- **不强制 4-bit MTP 头** — 头本身 ~3B active,可 4/6/8-bit;quant 脚本支持。

## 3. Architecture

### 3.1 代码分布

```
crates/
  qwen36-spec/          # NEW — Qwen3.6 family substrate
    src/lib.rs          # config / weight name resolution / MTP head metadata
    src/mtp.rs          # MTP draft head Rust-side glue + accept loop
  mlx-sys/
    src/mlx_mtp_draft_model.cpp   # NEW — sibling of mlx_dflash_draft_model.cpp,
                                  # 复用 SwiGLU / quantized_matmul / RMSNorm 原语
    src/lib.rs          # 加 mtp_* FFI 符号
  infer/
    src/metal/spec/     # NEW dir — 抽象 "draft source" 让 DFlash / MTP 共栈
      mod.rs
      dflash.rs         # 移过来(现在散在 sched 里)
      mtp.rs            # MTP entry,wrap qwen36-spec
    src/metal/serve.rs  # 加 --spec-type {none,dflash,mtp} 解析
scripts/
  bench_36_mtp.py       # NEW — 三档 bench(baseline / dflash 注入 / mtp)
  convert_qwen36_mtp_to_mlx_4bit.py  # NEW — 保留 mtp.* 的 MLX 4-bit quant
docs/
  experience/wins/2026-05-XX-metal-mtp-qwen36-<result>.md   # 成功后写
```

### 3.2 抽象边界 — `DraftSource` trait

```rust
// crates/infer/src/metal/spec/mod.rs
pub trait DraftSource {
    /// 给定 token 序列 + 主模型最后 hidden state,产出 K 个 draft tokens
    /// + 对应 logits(供 Leviathan-Chen ratio)。
    fn draft(
        &mut self,
        prefix_ids: &[u32],
        prefix_hidden: &mlx::Array,  // [seq, hidden]
        k: usize,
    ) -> DraftBundle;

    /// 一次 cycle 完成后通知(供 KV-state 回滚)
    fn rollback(&mut self, accepted: usize);
}

pub struct DraftBundle {
    pub tokens: Vec<u32>,        // K draft tokens
    pub q_logits: Vec<Vec<f32>>, // [K, vocab] draft logits q(x)
}
```

- `DFlashDraftModel` 实现 `DraftSource`:跑独立小 transformer 拿 logits
- `MtpDraftHead` 实现 `DraftSource`:复用 prefix_hidden + 跑单层 MoE 头拿 logits

主 sched loop 不需要知道是哪个 source,只 dispatch `accept_speculative()`。

### 3.3 与 DFlash 共存

| backend | dense model | MoE model |
|---|---|---|
| Metal DFlash | ✅ Qwen3 / Qwen3.5 5.9x | ❌ acceptance 太低,不上 |
| Metal MTP | (不上 — 没 MTP 头) | ✅ Qwen3.6 family |
| Metal none | ✅ fallback | ✅ fallback |

`--spec-type auto`:dense → DFlash;MoE 且 `mtp.*` weights 存在 → MTP;否则 none。

## 4. MTP 头权重装载

### 4.1 上游格式(`Qwen/Qwen3.6-35B-A3B`)

```
mtp.fc.weight                          # [hidden*2, hidden]  proj concat(h_prev, embed)
mtp.layers.0.input_layernorm.weight    # [hidden]
mtp.layers.0.post_attention_layernorm.weight  # [hidden]
mtp.layers.0.self_attn.q_proj.weight   # [hidden, hidden]    (GQA: q heads)
mtp.layers.0.self_attn.k_proj.weight   # [hidden, kv_dim]
mtp.layers.0.self_attn.v_proj.weight   # [hidden, kv_dim]
mtp.layers.0.self_attn.o_proj.weight   # [hidden, hidden]
mtp.layers.0.mlp.gate.weight           # [hidden, n_experts] router
mtp.layers.0.mlp.experts.gate_up_proj  # [n_experts, hidden, 2*ffn]
mtp.layers.0.mlp.experts.down_proj     # [n_experts, ffn, hidden]
mtp.norm.weight                        # final norm
mtp.lm_head.weight  (= shared embed?)  # 待定 — Qwen 多半 share
```

→ 结构 = 一个标准 Qwen3.6 decoder layer + 输入 fc 拼接。**复用 ARLE 现有 MoE
expert dispatch + GQA attention kernel**,只需新写 `MtpHead` 容器把它们组好。

### 4.2 Quant 路径

社区两个主流 MLX 4-bit(`mlx-community/Qwen3.6-35B-A3B-4bit`、
`unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit`)**都 strip 了 `mtp.*`**(2026-05-19 实测
both repos `mtp` keys = 0)。两条路:

1. **`scripts/convert_qwen36_mtp_to_mlx_4bit.py`(推荐,~1d 工作量)**
   - 输入:`Qwen/Qwen3.6-35B-A3B` BF16(71.9 GB)
   - 输出:ARLE MLX format with `mtp.*` 4-bit 量化保留
   - 复用 `mlx_lm.convert` 接口,加白名单不 strip MTP keys
   - 头本身 ~3B param × 4-bit = ~1.5 GB 额外磁盘

2. **直接消费 unsloth MTP-GGUF**(快速验证用,非长期路径)
   - `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` 已含 MTP heads
   - 需 ARLE Metal 走 GGUF 路径(已支持)+ MTP 头读 GGUF tensor map
   - 缺点:GGUF runtime 在 Mac 上比 MLX 慢 ~2x(实测 38 vs 86 tok/s baseline),
     抵消 MTP 收益。仅做 correctness baseline,不做 perf 决策。

### 4.3 安全检查(loader)

- safetensors index.json 里 grep `^mtp\.` 找到 → enable MTP path
- 找不到 → log warning,`--spec-type mtp` 降级为 none,不 hard fail
- shape mismatch → hard fail with actionable error(列出 expected vs got)

## 5. 解码循环算法

### 5.1 单 cycle(K=2, the M4 Pro sweet spot per llama.cpp bench)

```
状态:prefix_ids[..t],prefix_hidden[..t]

1. mtp.draft(prefix_ids, prefix_hidden, k=2)
   → tokens [d0, d1], q_logits [q0, q1]   (q = draft 分布)

2. 主模型 batched forward:
   input  = [prefix_last_id, d0, d1]      (拼 K+1 个 position)
   output = [p0_logits, p1_logits, p2_logits]   (p = target 分布)

3. Leviathan-Chen accept(per-position):
   pos i: accept_ratio = min(1, p_i[d_i] / q_i[d_i])
   sample u ~ U(0,1)
   accept if u < accept_ratio
   一旦 reject,break

4. On reject at position i:
   resid = max(p_i - q_i, 0)
   resid /= resid.sum()
   replacement = sample(resid)
   emit accepted_prefix + [replacement]

5. On full accept(都没 reject):
   bonus = sample(p_K)   ← 最后一个 position 的 free token
   emit accepted_prefix + [bonus]   → 共 K+1 = 3 tokens this cycle

6. KV cache:append accepted tokens,drop rejected slot
```

### 5.2 Temp=0 special case

- greedy:`d_i = argmax q_i`,`p_i[d_i]` 在 MTP 训练充分时 ≈ 1
- accept ratio 退化为 `1.0` 几乎所有 case → bit-identical to non-spec at temp=0
- correctness gate:**温度 0 下 MTP path 输出必须与 non-spec path bit-identical**

### 5.3 K 的自动选择

- 启动时跑 256-token micro-bench,扫 K ∈ {1,2,3,4},取 max tok/s
- 缓存到 `~/.cache/arle/mtp_k_<model_hash>.json`
- 也可 CLI override:`--spec-draft-n-max 2`

依据:llama.cpp 上 K=2 sweet spot 在 M4 Pro,K=6 反而 -24%(2026-05-19 实测)。
ARLE MLX runtime baseline 高 2x,sweet spot 可能不同 —— **必须 micro-bench**。

## 6. Phased implementation

| Phase | 内容 | 退出 license | 预计工作量 |
|---|---|---|---|
| **P0 — Weight verify** | 写 `convert_qwen36_mtp_to_mlx_4bit.py`,产出带 `mtp.*` 的 ARLE MLX 4-bit 包;在 mlx-lm 里手动 forward 一次 MTP head 验证 logits 合理 | weight load 成功 + draft logits top-1 与官方 BF16 余弦相似度 ≥ 0.98 | 1d |
| **P1 — Loader + DraftSource skel** | `crates/qwen36-spec` 新建,`MtpHead` 装载 + 单 token forward(K=1 退化);infer 加 `--spec-type mtp` 走 fallback when keys missing | `arle serve --backend metal --spec-type mtp` 单请求不 crash,k=1 输出 = baseline 输出 | 2d |
| **P2 — Full cycle K≥2** | accept loop + residual correction,KV 回滚;temp=0 bit-identical gate | bench 256-token 输出在 K=2/temp=0 下 bit-identical with no-spec;tok/s ≥ baseline | 3d |
| **P3 — Perf gate** | 跑 `scripts/bench_36_mtp.py`,对照 no-spec / DFlash / MTP × K∈{1,2,3,4} × temp∈{0,0.6} | **温度 0:≥ 100 tok/s;温度 0.6:≥ 110 tok/s**(M4 Pro 48GB 上) | 1d |
| **P4 — Win 文档 + matrix 更新** | `docs/experience/wins/2026-05-XX-metal-mtp-qwen36-<result>.md`;`docs/support-matrix.md` Qwen3.6 行从 "Beta load only" 升级 | 文档 ship + README/主页 bench row 替换 | 0.5d |

总:**~7.5 工作日**(单人,熟悉 mlx-sys 与 metal_serve)。

## 7. Performance gates

| Workload | 当前 baseline(2026-05-19) | P3 gate | Stretch |
|---|---:|---:|---:|
| 91-tok prompt / 256 out / temp 0 / single stream | 86.3 tok/s(MLX) | **≥ 100 tok/s** | ≥ 120 |
| 同上 / temp 0.6 / top_p 0.95 | 未测 | **≥ 110 tok/s** | ≥ 130 |
| 1k prompt / 256 out / temp 0 | 未测 | ≥ 90 tok/s | ≥ 105 |
| 4k prompt / 128 out / temp 0(prefill heavy)| 未测 | 不退化(≥ 85% baseline)| — |

gate 不达 → 不 land。fallback:文档化 MoE 的 MTP 在 M4 Pro 上的真实 ceiling
(可能就是 +10–15%),进 errors 而不是 wins。

## 8. Correctness gates

- **G1 bit-identical at temp=0**:同 seed 同 prompt,MTP path 与 non-spec path 输出
  token-for-token 一致,在 1000 prompt × 256 token sample 上。差异 → 算法 bug。
- **G2 distribution preserving at temp>0**:KL(MTP || non-spec)≤ 0.02 nats on
  10k token sample at temp=0.6。**Leviathan-Chen + residual correction 数学上保证 = 0**,
  实测看 floating-point 噪声。
- **G3 acceptance rate**:K=2 下 acceptance ≥ 70%(MoE 下限,参 vLLM 经验)。
  < 70% → MTP 头权重装载有问题或 quant 损坏。

每个 gate 写成 pytest in `tests/integration/test_metal_mtp_qwen36.py`,P3 必跑。

## 9. 风险与开放问题

### R1 — M4 Pro 内存带宽天花板可能本质上压死 MTP

- baseline 已吃 78% 带宽。MTP 的 verify 是 batched K+1 forward,**单步带宽需求
  ≈ (K+1)× 单步** = 3x for K=2。
- 若 batched forward 不能 amortize(MoE expert routing 在 batched setting 下可能
  load 更多 experts),净收益可能 < 10%。
- **缓解**:P3 实测决定。若 < 10%,写 errors doc 说明物理上限,不强行 land。

### R2 — MoE MTP 头的 expert routing 是否需要跟主模型 sync

- Qwen3.6 MTP 头有自己的 router(`mtp.layers.0.mlp.gate.weight`),独立 select
  experts。这导致 KV 状态分支需要谨慎处理 — 但因为 MTP 头 forward 是独立 layer,
  不影响主模型 KV。
- 风险:experts share weight pool — 同 cycle 主模型 + MTP 头同时访问 expert 权重,
  Metal cmd buffer 可能 serialize。
- **缓解**:P2 时插入 metal trace,确认 expert kernel 不互锁。

### R3 — Quant 脚本生态风险

- 自己写 MLX 4-bit MTP-preserving quant 是 fresh code,可能有 subtle bug。
- 第三方(unsloth/mlx-community)未来可能补上 MTP-preserving 版本,届时直接消费即可。
- **缓解**:P0 加 cosine-similarity gate vs 官方 BF16(≥ 0.98),数学验证 quant 没坏。

### R4 — Qwen 后续版本 MTP 结构变化

- 当前 (`mtp.layers.0.*`)是单层。Qwen 4.0 / 4.1 可能改成多层或 EAGLE 风。
- **缓解**:`MtpHead` 走 layer count 参数化(`mtp.layers.{0..N}`),不 hardcode。

### Open questions(需 owner 决策)

- Q1:K 自动 micro-bench 跑哪个 prompt?(代码 / 通用文本两种 acceptance 差异大)
- Q2:CUDA 端原生 MTP 是否同样 design?(假设是,但本 doc 不覆盖实现细节)
- Q3:multi-tenant serve 下,MTP cycle 是否 per-slot 独立 K?

## 10. Bench plan(P3)

```bash
# 三档对照
scripts/bench_36_mtp.py \
  --model Qwen3.6-35B-A3B-mtp-mlx-4bit \
  --modes none,dflash,mtp \
  --k-sweep 1,2,3,4 \
  --temps 0.0,0.6 \
  --prompt-len 91,1024,4096 \
  --output-tokens 256 \
  --n-runs 5 \
  --out bench-output/2026-05-XX-metal-mtp-qwen36/
```

输出格式与 `bench-output/2026-04-20-metal-qwen36-compiled-moe-quick/headline_table.md`
对齐,贴 wins doc 直接用。

## 11. Migration / rollback

- 默认 `--spec-type none`(unchanged behavior),用户主动开 `mtp` 才走新 path
- weight 装载失败自动降级 + log warning,不 hard fail
- 若 P3 perf gate 没过,代码可保留为实验性 path(`#[cfg(feature = "experimental-mtp")]`),
  不进 release artifacts

## 12. SOLID 自检(per AGENTS.md)

- **S — Single responsibility**:`MtpHead` 只做 draft;`accept loop` 在 `infer/spec/`
  共享;quant 脚本独立 → ✅
- **O — Open/closed**:`DraftSource` trait 允许未来加 EAGLE / Medusa 不动 sched → ✅
- **L — Liskov**:DFlash 和 MTP 都实现同一 trait,sched 不区分 → ✅
- **I — Interface segregation**:DraftSource 只暴露 draft + rollback,不耦合 KV
  细节 → ✅
- **D — Dependency inversion**:sched 依赖 trait 而非 concrete struct → ✅

### Gap 自检

| Gap | 现状 |
|---|---|
| 无 owner | 待指定 |
| 工作量估算无 buffer | 已含 ~30% 给 Metal 调试 |
| CUDA 接口契约未写 | 显式列入 Out of scope(§2.2);Phase 5+ |
| multi-tenant 行为 | 列入 Open question(Q3) |
| 自家 quant 脚本可能 quality 退化 | P0 gate 用 cosine-sim ≥ 0.98 兜底 |

---

## Appendix A — 实测数据来源

- 2026-05-18 ad-hoc bench:MLX 86.3 / ARLE serve 85.6 tok/s,详见本会话 transcript +
  `README.md#status-at-a-glance` 更新
- 2026-05-19 ad-hoc bench:llama.cpp baseline 38.4,MTP n_max=2 44.3,n_max=6 29.3
- 历史:`bench-output/2026-04-20-metal-qwen36-compiled-moe-quick/headline_table.md`
  (conc1 42.2 tok/s — 旧 GuideLLM 路径,与今天 streaming bench 不可直接对比)

## Appendix B — 参考实现

- vLLM `vllm/v1/spec_decode/mtp_proposer.py`(CUDA,算法权威)
- SGLang MTP path,`--speculative-algo NEXTN`(0.5.10+)
- MTPLX(MLX-Swift native,Qwen3.6-27B 实测 2.24x,Apple Silicon 端最接近的实现参考)
- llama.cpp PR #19493(2026-05-16 merge,`--spec-type draft-mtp`,GGUF 端权威)
