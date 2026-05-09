---
title: M_rope-yarn-scaling — RoPE YARN/NTK/PI 实现 plan(long-ctx unblocker)
date: 2026-05-09
type: plan
status: design-ready
audience: codex (substrate pickup)
---

# M_rope-yarn-scaling — RoPE scaling 缺失,blocks all long-ctx > native train

> Phase 0 source-grep audit:ARLE **未实现** RoPE scaling(YARN / NTK-aware /
> PI),全用 `rope_theta` base。Qwen3-4B 32k native train 已 OK,但
> 32k-128k leadership project Phase 2-4 + Qwen3.6 260k context **都 blocked**
> 此处。

## 1. Phase 0 audit

### 当前实现(grep findings)

ARLE RoPE 只 carry `rope_theta: f32` field through 整 stack:
- `crates/qwen3-spec/`:`rope_theta`(no scaling field)
- `crates/qwen35-spec/`:同
- `infer/src/gguf.rs:1063`:`pub rope_theta: f32`
- `infer/src/model/qwen3/weights.rs:448`:`precompute_rope(&ctx, head_dim, rope_cache_len, config.rope_theta)`
- `infer/src/backend/metal/qwen35.rs:1383`:`config.rope_theta as f32`
- DFlash `rope_theta` 同

→ **NO YARN / NTK / PI / linear interpolation 路径**,全 vanilla RoPE。

### 影响范围

| 场景 | 当前最大 ctx | RoPE scaling 后 |
|------|------------:|----------------:|
| Qwen3-4B(32k native train)| **32k**(限 native)| 128k(factor=4 YARN)/ 256k(factor=8)|
| Qwen3.6 35B-A3B(32k native)| **32k**(限 native)| 64k-260k(YARN factor=2-8)|
| 32k-128k leadership project Phase 2-4 | **blocked at 32k** | **可启** |
| Qwen3.6 260k context 用户课题 | **blocked at 32k** | **可启** Phase A→D |

**Scope**:全 long-ctx work item 都 blocked。**ROI 高**。

## 2. 实现 plan(LOC 估计)

### 2.1 Config 接入(40-60 LOC)

```rust
// crates/qwen3-spec + qwen35-spec
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RopeScalingConfig {
    #[serde(rename = "yarn")]
    Yarn {
        factor: f32,
        original_max_position_embeddings: usize,
        beta_fast: Option<f32>,        // default 32
        beta_slow: Option<f32>,        // default 1
        attention_factor: Option<f32>, // default auto
        mscale: Option<f32>,           // default 1
    },
    #[serde(rename = "linear")]
    Linear { factor: f32 },
    #[serde(rename = "ntk_aware")]
    NtkAware { factor: f32 },
    #[serde(rename = "pi")]
    PositionInterpolation { factor: f32 },
}

pub struct ModelConfig {
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScalingConfig>,
    // ...
}
```

### 2.2 inv_freq 计算(70-100 LOC)

YARN 算法核心:linear interpolation low freq + NTK-aware extrap high freq +
attention factor 补偿:

```rust
fn yarn_inv_freq(
    head_dim: usize,
    base: f32,
    factor: f32,
    original_max_pos: usize,
    beta_fast: f32,
    beta_slow: f32,
) -> Vec<f32> {
    // 1. 基础 inv_freq = 1.0 / (base ** (i/dim) for i in 0..dim/2)
    let dim = head_dim;
    let mut inv_freq = (0..dim/2)
        .map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32))
        .collect::<Vec<_>>();
    
    // 2. low / high frequency 分界(per YARN paper §3.2)
    let low_freq_factor = (original_max_pos as f32) / (beta_fast * std::f32::consts::TAU);
    let high_freq_factor = (original_max_pos as f32) / (beta_slow * std::f32::consts::TAU);
    
    // 3. 每个 dim mix linear / NTK-extrapolation
    for (i, freq) in inv_freq.iter_mut().enumerate() {
        let wavelen = std::f32::consts::TAU / *freq;
        let smooth = ((wavelen - low_freq_factor).max(0.0) / (high_freq_factor - low_freq_factor)).clamp(0.0, 1.0);
        let scaled = *freq / factor;  // linear (low freq)
        let extrap = *freq;            // NTK extrapolation (high freq)
        *freq = scaled * (1.0 - smooth) + extrap * smooth;
    }
    inv_freq
}

fn yarn_attention_factor(factor: f32, mscale: f32) -> f32 {
    // YARN paper §3.4 attention scale 补偿
    1.0 + 0.1 * mscale * (factor.ln())
}
```

### 2.3 集成到 precompute_rope path(40-60 LOC)

```rust
// infer/src/model/qwen3/weights.rs:448 周围
let inv_freq = match config.rope_scaling.as_ref() {
    None => default_rope_inv_freq(config.head_dim, config.rope_theta),
    Some(RopeScalingConfig::Yarn { factor, original_max_position_embeddings, beta_fast, beta_slow, .. }) => 
        yarn_inv_freq(config.head_dim, config.rope_theta, *factor, *original_max_position_embeddings,
                      beta_fast.unwrap_or(32.0), beta_slow.unwrap_or(1.0)),
    Some(RopeScalingConfig::Linear { factor }) => 
        linear_inv_freq(config.head_dim, config.rope_theta, *factor),
    Some(RopeScalingConfig::NtkAware { factor }) =>
        ntk_aware_inv_freq(config.head_dim, config.rope_theta, *factor),
    // ...
};
let cos_cache = compute_cos_cache(rope_cache_len, &inv_freq);
let sin_cache = compute_sin_cache(rope_cache_len, &inv_freq);
```

attention_factor 应用到 attention scores(乘进 softmax 前的 logits):
```rust
// 在 prefill / decode attention path 加 attention_factor
attn_scores *= attention_factor;
```

### 2.4 Metal 同步(40-60 LOC)

`infer/src/backend/metal/qwen35.rs:1383` + `dflash.rs:522` 接入相同 RoPE
scaling path。Metal-side 无独立 RoPE 内核,inv_freq 是 host 计算 → 上传 device。

### 2.5 Tests + greedy_consistency(30-50 LOC)

- Unit test:YARN factor=2 inv_freq 数值 vs 上游(transformers / mlx 参考实现)
- E2E test:`cargo test --release greedy_consistency -- --nocapture` 同 prompt
  rope_scaling=None vs YARN factor=1 应数值等价(factor=1 是 noop)
- 长 ctx smoke:Qwen3-4B 64k context with YARN factor=2 不 panic + greedy
  decode 输出 reasonable

### 2.6 总 LOC 估计

| 子 task | LOC |
|---------|-----|
| Config 接入(qwen3-spec + qwen35-spec)| 40-60 |
| YARN inv_freq compute | 70-100 |
| Linear / NTK / PI inv_freq compute | 30-50 |
| precompute_rope 集成 + attention_factor | 40-60 |
| Metal 同步 | 40-60 |
| Tests | 30-50 |
| **总** | **250-380 LOC** |

→ codex pickup(超过 Claude < 100 LOC 上限)。

## 3. License-or-kill criteria

| 维度 | PASS | KILL |
|------|------|------|
| greedy_consistency rope_scaling=None vs factor=1 数值等价 | ✓ | 不等价 → factor=1 应是 noop |
| Qwen3-4B 32k native test e2e PASS | ✓ no regression | regression → 改 contaminate vanilla path |
| Qwen3-4B 64k YARN factor=2 smoke decode 输出 valid token | ✓ | gibberish / repetition → YARN impl bug |
| Unit test YARN inv_freq vs transformers 参考 | < 1e-4 deviation | larger → algorithm bug |
| LOC budget | ≤ 380 | > 500 split commits |

## 4. Phase 顺序 + 依赖

| Phase | 内容 | 依赖 | LOC |
|-------|------|------|-----|
| **1** | Config + YARN inv_freq impl + unit tests | (none) | 100-150 |
| **2** | precompute_rope 集成 + e2e PASS | Phase 1 | 80-120 |
| **3** | Metal 同步 + Qwen3.6 long-ctx smoke | Phase 2 + Mac access | 70-110 |
| **4** | Bench Qwen3-4B 32k vs 64k YARN tok/s + PPL | Phase 2 + GPU access | 0(bench only)|

→ Phase 1-2 可在 CUDA 机器上做 + verify(Qwen3-4B test fixtures 已有)。
→ Phase 3-4 需 Mac OR Linux+GPU bench。

## 5. 不在本 axis(scope discipline)

- ❌ FP8 KV cache Metal 实现(separate axis,#33 Phase 0a 之后)
- ❌ W4 KV cache impl(separate axis,可能 #33 之后才需)
- ❌ Sliding window attention(可选,vanilla GQA 通常足以撑长 ctx if memory enough)
- ❌ Adaptive KV cache(StreamingLLM/H2O,research-track,defer)

## 6. ROI

unblocks:
- 32k-128k leadership project Phase 2-4(Qwen3-4B 128k YARN factor=4)
- Qwen3.6 260k 用户课题 Phase B-D
- 任何 future model > 32k native train ctx 都 ready

cost:**250-380 LOC codex,1 week wall-clock**(含 multi-platform sync + bench)

## 7. Cross-references

- 调用方 long-ctx project:`docs/projects/2026-04-30-longctx-32k-128k-leadership.md`
- 调用方 Qwen3.6 260k 课题:`docs/research/2026-05-09-qwen36-35b-a3b-260k-context-feasibility.md`
- 当前 RoPE 实现:`infer/src/model/qwen3/weights.rs:448`,`infer/src/backend/metal/qwen35.rs:1383`
- YARN paper:Peng et al. 2023,`https://arxiv.org/abs/2309.00071`(reference only,不 fetch)
- 上游 reference impl:`transformers` 库 `LlamaYarnRotaryEmbedding`(public,可 cross-check 数值)

## 8. 状态

RoPE scaling = long-ctx hard prerequisite,ARLE 完全未实现。M_rope-yarn-scaling
plan ready for codex pickup。LOC 250-380 / 1 week wall-clock。**Phase 1-2 可在
CUDA 机器上 verify**,Phase 3 Metal 同步 + Qwen3.6 long-ctx smoke 需 Mac。
