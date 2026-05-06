# DSv4 Small-Scale Full-Method Repro Plan (Single 4070 Ti SUPER 16 GiB)

**Status:** scoping (2026-05-07)
**Owner:** ckl
**Hardware:** 1 × RTX 4070 Ti SUPER 16 GiB · CUDA 13.2 · cuDNN 9.21 · Linux · `.venv` at `/home/ckl/projects/arle/.venv`
**Goal:** **From-scratch pre-train** of a ~1 B-parameter model that is a 1:1 architectural / methodological copy of DeepSeek-V4 (Apr 24, 2026 release), shrunk to fit on one 16 GiB consumer GPU. **Not** distill, **not** SFT, **not** LoRA — a real cold-path pre-train.
**Companion plan:** [`2026-05-05-deepseek-v4-small-substrate.md`](2026-05-05-deepseek-v4-small-substrate.md) covers the runtime substrate (MLA-era V3 fixtures + nano training driver). This plan is scoped narrower (one SKU, V4-era) and deeper (full method, all data sources, all hyper-params).

---

## 0. TL;DR — chosen target

| | DSv4-Pro reference | **dsv4-mini** (this plan) |
|---|---|---|
| total params | 1.6 T | **~1.05 B** |
| activated params | 49 B | **~330 M** |
| `hidden_size` | 7168 | **1024** |
| `num_hidden_layers` | 61 | **24** |
| `num_attention_heads` | 128 | **16** |
| `num_key_value_heads` (GQA) | 1 | **1** |
| `head_dim` (V4 unified) | 512 | **128** |
| `q_lora_rank` | 1536 | **512** |
| `o_lora_rank` / `o_groups` | 1024 / 16 | **512 / 8** |
| `qk_rope_head_dim` | 64 | **32** |
| `n_routed_experts` / per-tok | 384 / 6 | **32 / 4** |
| `n_shared_experts` | 1 | **1** |
| `moe_intermediate_size` | 3072 | **1408** |
| `num_nextn_predict_layers` (MTP) | 1 | **1** |
| `index_n_heads` / `index_topk` / `index_head_dim` | 64 / 1024 / 128 | **8 / 256 / 64** |
| `num_hash_layers` | 3 | **3** |
| `sliding_window` | 128 | **128** |
| `compress_ratios` (CSA=4, HCA=128, SWA=0) | 61 entries | **24 entries** (P0/P1=HCA, then alternate CSA/HCA) |
| `vocab_size` | 129 280 | **49 152** (3-byte BPE; smaller, see §3.1) |
| `max_position_embeddings` (YaRN base 65k → ×16) | 1 048 576 | **65 536** (YaRN base 8 192 → ×8 = 65 k) |
| pretrain tokens | 32 T | **40 B** (~38× Chinchilla under-trained, per "method ≠ score" goal) |
| precision | FP4 expert / FP8 dense | **BF16 master + FP8 dense matmuls (TE)** — FP4 deferred (see §6.3) |
| optimizer | Muon (hidden 2D) + AdamW (norm/embed) | **same** |

**Memory accounting** for `dsv4-mini` at `seq=4096`, `micro-batch=1`, BF16 master + FP8 fwd, ZeRO-1, gradient checkpointing on every transformer block:

```
weights (BF16, 1.05 B × 2 B)         ~ 2.10 GiB
master fp32 weights (Muon hidden)    ~ 4.20 GiB
optimizer state (Muon momentum fp32) ~ 4.20 GiB
gradients (BF16)                     ~ 2.10 GiB
activations (ckpt: 1 layer worth)    ~ 0.80 GiB  (4096 × 1024 × 24 × 2 B / 24 stored)
KV + attention scratch (CSA index)   ~ 0.50 GiB
TE / cuBLAS workspace + framework    ~ 1.00 GiB
─────────────────────────────────────────────
≈ 14.9 GiB                            margin ≈ 1.1 GiB
```

If `Muon master + momentum` is moved to **CPU offload** (DeepSpeed ZeRO-Offload, the only way Muon-on-1B-on-16GB is actually safe), the on-GPU state drops by ~8.4 GiB and we have **~7 GiB headroom**, enough to push `seq=8192` or `micro-batch=2`. **Decision: use ZeRO-2 + optimizer offload** (see §4.3).

---

## 1. DSv4 architectural ground truth

All numbers below are from the official `config.json` of `deepseek-ai/DeepSeek-V4-Pro` and `DeepSeek-V4-Flash-Base` (released 2026-04-24), the Hugging Face DSv4 blog post, the technical report PDF (`deepseek-V4-model-card-EN.pdf`), and the Muon scalability paper. Items the public docs do not pin down are tagged **[假设]**.

### 1.1 Hybrid attention — **the V4 delta**

DSv4 **abandons MLA**. The V3-era `kv_a_proj_with_mqa` / `kv_b_proj` / `kv_lora_rank` triplet is gone from the V4 config. The new shape:

- **Single-KV-head GQA**: `num_key_value_heads = 1`. All KV is shared across all Q heads — radical KV compression at the head-count axis.
- **Q-LoRA**: `q_lora_rank = 1536` (Pro) / 1024 (Flash). Q is `hidden → q_lora_rank → q_proj_dim` two-stage. Same trick V3 used for q only.
- **O-LoRA with grouping** (NEW in V4): `o_lora_rank = 1024`, `o_groups = 16` (Pro) / 8 (Flash). Output projection is split into `o_groups` parallel rank-`o_lora_rank` factorisations, then summed. Saves params on the largest projection.
- **Unified `head_dim = 512`** (huge): each attention head writes a 512-wide value. Combined with `num_key_value_heads = 1` and `o_groups`, V4's per-head FLOP profile is unlike V3.
- **Per-layer attention type via `compress_ratios[L]`**:
  - `0` → **Sliding-Window Attention (SWA)** with window=128, used only on layer 0/1 of Flash (warm-up layers).
  - `4` → **Compressed Sparse Attention (CSA)**: stride-4 compressor (overlap pool of 8 raw KV → 1 compressed entry) + a **Lightning Indexer** that selects top-`index_topk` compressed positions per query, giving sparse-but-dense-quality attention.
  - `128` → **Heavily Compressed Attention (HCA)**: stride-128 non-overlap compressor, dense attention over the highly compressed stream. Used on layer 0/1 of Pro and on alternating layers of both Pro and Flash.
- **Lightning Indexer config**: `index_n_heads=64`, `index_head_dim=128`, `index_topk=1024` (Pro) / 512 (Flash). The indexer is a small attention-like network that scores each compressed KV vs each query and picks top-k.
- **Rotary**: `rope_theta=10_000` for the dense path; `compress_rope_theta=160_000` for the compressed/HCA path (different base because compressed positions span very different stride). YaRN scaling: `factor=16`, `original_max_position_embeddings=65_536`, `beta_fast=32`, `beta_slow=1`. Effective ctx = `65_536 × 16 = 1_048_576`.
- **Hashed positions**: `num_hash_layers=3`. **[假设]** Used inside the Lightning Indexer for stable long-range positional encoding; the report describes it as "hashed compressor positions" but exact hashing scheme is not in the public model card.
- **Sliding window everywhere**: `sliding_window=128` is layered *on top* of the compressed paths to preserve local detail.

KV-cache savings vs V3 at 1 M context: 10% of V3's KV, 27% of V3's per-token FLOPs.

### 1.2 MoE — minor delta from V3

- `n_routed_experts = 384` (Pro) / 256 (Flash); V3 was 256.
- `num_experts_per_tok = 6` (V3 used 8). Top-k drop is part of the V4 efficiency push.
- `n_shared_experts = 1` (unchanged).
- `moe_intermediate_size = 3072` (Pro) / 2048 (Flash); V3 was 2048.
- `routed_scaling_factor = 2.5` (Pro) / 1.5 (Flash) — applied to weighted sum of routed experts, V3 also used 2.5.
- `topk_method = "noaux_tc"` — **auxiliary-loss-free** load balancing with router-bias adjustment, identical mechanism to V3.
- `scoring_func = "sqrtsoftplus"` (NEW in V4). Replaces V3's plain `sigmoid` gate with `sqrt(softplus(score))`, which is positive, monotone, and grows like `√x` for large positive `x`. **[假设, 高置信度]** This dampens runaway router scores and pairs with the no-aux-loss bias adjustment to keep utilisation flat without loss term pressure.
- `norm_topk_prob = true` — top-k weights re-normalised to sum to 1.
- **No expert grouping in V4** — V3 had `n_group=8 / topk_group=4`; V4 drops it (router bias keeps routing balanced without group-limited pre-selection).

### 1.3 Manifold-Constrained Hyper-Connections (mHC)

DSv4 replaces plain residual `x + f(x)` with a learned **Birkhoff-polytope-projected mixing matrix** between `n_hc=4` parallel residual streams. Knobs from config:

- `hc_mult = 4` — number of hyper-connection streams per layer.
- `hc_sinkhorn_iters = 20` — Sinkhorn iterations to project the mixing matrix onto the Birkhoff polytope (doubly-stochastic non-negative matrices) at each forward pass.
- `hc_eps = 1e-6` — Sinkhorn numerical eps.

Per the technical report blog: mHC strengthens signal propagation and "preserves expressivity" vs vanilla residual. **Implementation cost on 1 B-scale: negligible** (Sinkhorn on a 4×4 matrix per layer per forward).

### 1.4 MTP

- `num_nextn_predict_layers = 1` (Pro and Flash). V4 keeps a single MTP head — exactly like V3-style MTP. The head is a full transformer block plus `eh_proj` cross-projection from token-`t` hidden into token-`t+1` predictor stream.
- Used during pre-training as an auxiliary loss (predict `t+1` and `t+2` jointly), discarded at inference unless used as a speculative-decoding proposer.

### 1.5 Quantisation layout

- **Expert weights**: FP4 (`expert_dtype = "fp4"` on Pro, `fp8` on Flash). Block-wise scale (`weight_block_size = [128, 128]`), `scale_fmt = "ue8m0"` (8-bit unsigned exponent — power-of-two scales).
- **Non-expert weights**: FP8 E4M3 forward (`fmt = "e4m3"`), with **dynamic per-tensor activation scaling** (`activation_scheme = "dynamic"`).
- The `quantization_config` block in `config.json` is the source of truth.

### 1.6 Optimizer & training procedure

- **Muon** for all 2-D hidden weights (attention/MLP/MoE projections). Reference Muon hyperparams (Moonlight paper): hidden lr ≈ 0.02, weight_decay = 0.1, momentum = 0.95, Nesterov, ns_steps = 5.
- **AdamW** for embedding, LayerNorm/RMSNorm scales, biases, router gate. lr ≈ 3e-4 typical.
- LR schedule: linear warmup (~2 k steps) → cosine decay to 0.1× peak.
- **Sequence-length curriculum**: `4 K → 16 K → 64 K → 1 M`. Dense attention for the **first 1 T tokens at 4 K/16 K**, then sparse attention (CSA/HCA) is enabled at the 64 K extension. We mirror this curriculum at small scale (§4.4).
- Total tokens: 32 T (Flash) / 33 T (Pro). Composition: math, code, web, long documents, scientific papers, multilingual. Auto-generated/templated content removed.
- Post-training:
  1. **Specialist Training**: SFT on domain data → GRPO RL.
  2. **On-policy distillation**: student matches per-expert teacher distribution.

For the small-scale repro, **we ship pre-train + a token-budget GRPO pass on math/code only** (post-training optional, see §6).

---

## 2. dsv4-mini — exact config

Single `config.json` (compatible with `transformers` ≥ 4.57.1; same loader path that V4-Pro/Flash use):

```json
{
  "architectures": ["DeepseekV4ForCausalLM"],
  "model_type": "deepseek_v4",
  "torch_dtype": "bfloat16",

  "vocab_size": 49152,
  "hidden_size": 1024,
  "num_hidden_layers": 24,
  "num_attention_heads": 16,
  "num_key_value_heads": 1,
  "head_dim": 128,
  "hidden_act": "silu",
  "swiglu_limit": 10.0,

  "q_lora_rank": 512,
  "o_lora_rank": 512,
  "o_groups": 8,
  "qk_rope_head_dim": 32,

  "n_routed_experts": 32,
  "n_shared_experts": 1,
  "num_experts_per_tok": 4,
  "moe_intermediate_size": 1408,
  "routed_scaling_factor": 1.5,
  "norm_topk_prob": true,
  "scoring_func": "sqrtsoftplus",
  "topk_method": "noaux_tc",

  "index_n_heads": 8,
  "index_head_dim": 64,
  "index_topk": 256,
  "num_hash_layers": 3,
  "sliding_window": 128,
  "compress_ratios": [128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0],

  "hc_mult": 4,
  "hc_sinkhorn_iters": 20,
  "hc_eps": 1.0e-6,

  "num_nextn_predict_layers": 1,

  "max_position_embeddings": 65536,
  "rope_theta": 10000,
  "compress_rope_theta": 160000,
  "rope_scaling": {
    "type": "yarn",
    "factor": 8,
    "original_max_position_embeddings": 8192,
    "beta_fast": 32,
    "beta_slow": 1
  },

  "rms_norm_eps": 1.0e-6,
  "initializer_range": 0.02,
  "tie_word_embeddings": true,
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 1,

  "expert_dtype": "fp8",
  "quantization_config": {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "scale_fmt": "ue8m0",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128]
  }
}
```

**Param count breakdown** (BF16, tied embeddings):

| Block | Per-layer | Layers | Subtotal |
|---|---|---|---|
| Embedding (tied) | `49 152 × 1024` | 1 | 50 M |
| Per-layer attention (q-LoRA + 1 KV head + o-LoRA × 8 groups + RMSNorms + indexer share) | ~7.5 M | 24 | 180 M |
| Per-layer MoE: 32 routed × `2 × 1024 × 1408 + 1024` + 1 shared × same + router | ~33 M routed-only-counted-once | 24 | 790 M |
| MTP head (1 transformer block + eh_proj) | ~33 M | 1 | 33 M |
| Final RMSNorm + tied LM head | – | – | 0 |
| **Total** | | | **≈ 1.05 B** |
| **Activated per token** (top-4 of 32 + shared + dense) | | | **≈ 330 M** |

Activated-param ratio matches DSv4-Flash's 13/284 ≈ 4.6%; ours is 330/1050 ≈ 31% because at small scale `n_routed_experts` collapses faster than `top-k`. Not 1:1 in ratio but **architecturally identical** in the routing/sharing scheme — that is the goal.

---

## 3. Training data — China-accessible sources only

Token target: **40 B** (compute-feasible at 4070 Ti S; under-trained vs Chinchilla — fine, the goal is method validation).
Composition target (mirrors DSv4 statements where they pin down %, otherwise [假设]):

| Slice | Target tokens | Target % | Source(s) |
|---|---|---|---|
| English web | 16 B | 40% | FineWeb (mirror) |
| Chinese web (edu-grade) | 12 B | 30% | OpenCSG Chinese-Fineweb-Edu V2.2 (ModelScope) |
| Code | 8 B | 20% | The-Stack-v2-dedup subset (mirror) |
| Math | 2 B | 5% | OpenWebMath + DeepSeek-Math corpus (mirror) |
| Long-doc / scientific | 1.5 B | 3.75% | RedPajama-arXiv subset (mirror) |
| Multilingual (residual) | 0.5 B | 1.25% | OPUS subset (mirror, optional) |

### 3.1 Tokenizer

- **DeepSeek tokenizer is open** (`tokenizer.json` is published with every DSv3/DSv4 release under MIT license, e.g. `deepseek-ai/DeepSeek-V4-Pro/tokenizer.json`). 129 280 vocab.
- For dsv4-mini we **train a smaller 49 152-vocab BPE on a 5 B-token mix** of the same data sources, using `crates/train/examples/build_bpe_tokenizer.rs` (already in tree, used by the existing `pretrain-dsv4` nano flow). 49 152 saves ~80 MB embedding params at our scale and is empirically fine for ~1 B models.
- **[假设]** If 5 B tokens is too slow to tokenize on a single CPU before training starts, fall back to **the official DSv4 tokenizer**, accept the 80 MB embed overhead. Pull command:
  ```bash
  modelscope download --model deepseek-ai/DeepSeek-V4-Pro \
    --include 'tokenizer.json' 'tokenizer_config.json' \
    --local_dir ./tokenizer_dsv4
  ```

### 3.2 Concrete data sources + commands

All commands assume `pip install modelscope huggingface_hub` and `export HF_ENDPOINT=https://hf-mirror.com` set globally for any HF fallback.

**(A) FineWeb (English web, EDU-filtered subset)** — Apache-2.0, ~1.3 T tokens for the EDU sample (we sub-sample 16 B).
Primary mirror: `hf-mirror.com`. ModelScope mirror also exists at `AI-ModelScope/fineweb-edu`.
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download HuggingFaceFW/fineweb-edu \
  --repo-type dataset \
  --include "data/CC-MAIN-2024-1*/*.parquet" \
  --local-dir ./data/fineweb-edu
```
Sub-select ~16 B tokens (~80 GB on disk parquet; tokens/byte ≈ 0.2 for English).

**(B) Chinese-Fineweb-Edu V2.2** — OpenCSG Community License + Apache-2.0, 188 M docs ≈ 420 B tokens.
ModelScope (no proxy needed in CN):
```bash
modelscope download --dataset opencsg/chinese-fineweb-edu-v2 \
  --local_dir ./data/zh-fineweb-edu
```
Sub-sample to 12 B tokens. Sources rolled in: WuDao, CCI3, Wanjuan, michao, ChineseWebText.

**(C) BAAI IndustryCorpus2** — MIT-style, 1 TB ZH + 2.4 TB EN cleaned multi-industry. Useful for the long-doc / scientific slice and to backfill code/math if (D)/(E) under-deliver.
```bash
modelscope download --dataset BAAI/IndustryCorpus2 \
  --local_dir ./data/industry-corpus-2
```

**(D) The-Stack-v2-dedup** (code) — bigcode license (per-language opt-in checked upstream). ~775 B tokens dedup. Sub-sample python/rust/c/cpp/go/js/ts → 8 B.
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download bigcode/the-stack-v2-dedup \
  --repo-type dataset \
  --include "data/python/*" "data/rust/*" "data/c/*" "data/cpp/*" "data/go/*" "data/javascript/*" "data/typescript/*" \
  --local-dir ./data/stack-v2
```

**(E) OpenWebMath + DeepSeekMath corpus** (math).
- OpenWebMath: 14.7 B tokens, ODC-By 1.0.
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  huggingface-cli download open-web-math/open-web-math --repo-type dataset \
    --local-dir ./data/openwebmath
  ```
- DeepSeekMath corpus public release (if available):
  ```bash
  modelscope download --dataset deepseek-ai/DeepSeekMath-Corpus \
    --local_dir ./data/deepseekmath || echo "[fallback] use OpenMathInstruct via hf-mirror"
  ```

**(F) RedPajama-arXiv** (long-doc / scientific) — Apache-2.0, ~28 B tokens.
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download togethercomputer/RedPajama-Data-1T --repo-type dataset \
  --include "arxiv/*" --local-dir ./data/redpajama-arxiv
```

**Disk budget (raw parquet/jsonl, before pre-tokenisation):** ~600 GB.
**Disk budget (tokenised, packed mmap binaries):** ~80 GB (uint32 token ids at 4 B/token, 40 B × 4 = 160 GB; with `bfloat16` packing or uint16 if vocab ≤ 65 535 → 80 GB).

### 3.3 Pre-processing

- **Pack to fixed `seq_len=4096` with no cross-doc attention leakage** (use `attn_segment_ids` in TE so packed sequences stay correct under causal mask). This matches V4's pack-then-train strategy.
- **De-dupe across the 6 slices** with MinHash-LSH at 5-gram level (see `datatrove` library, also on hf-mirror); skip if compute-cost prohibitive — these datasets are individually deduped.
- **Quality filter on Chinese slice**: keep only OpenCSG quality-score ≥ 3 (4-bin scale; cuts ~40% of bytes, retains best edu content).

---

## 4. Training pipeline

### 4.1 Framework choice

**Megatron-LM fork with DSv3 patches → patch up to V4** is the most direct path:
- `deepseek-ai/DeepSeek-V3` repo on GitHub publishes their training scripts (Megatron-LM-based).
- For V4, the open `glmnes/DeepSeek-V4` GH repo and **NVIDIA-NeMo/Automodel** (`docs/guides/llm/dsv4-flash.md`) both have V4-specific fork patches.
- **Recommendation:** clone NeMo-Automodel, point at our `dsv4-mini/config.json`, switch optimizer to Muon (NeMo has a Muon impl since 2026-Q1). NeMo handles ZeRO-2, FP8 via TransformerEngine, sequence packing, and gradient checkpointing out of the box.

Backup framework: **vanilla DeepSpeed + transformers `DeepseekV4ForCausalLM`**. The HF impl exists (transformers ≥ 4.57.1). Slower than NeMo for FP8 but lower cognitive load for a single-GPU run.

### 4.2 FP8 / FP4 strategy

- **FP8 for all dense BF16 → FP8 weight + FP8 forward matmul** through TransformerEngine `Linear`/`LayerNorm` modules. Dynamic per-tensor activation scaling (matches `activation_scheme: "dynamic"`).
- **FP4 expert weights**: Ada Lovelace (RTX 4070 Ti S, sm_89) does **not** have hardware FP4 tensor cores (those are Blackwell sm_100+ only). **Decision: train experts in FP8 for dsv4-mini**, label this as a known divergence from V4-Pro (which uses FP4). Set `expert_dtype: "fp8"` (matches V4-Flash, not Pro).
- **Master weights and Muon momentum in FP32** (Muon needs orthogonalisation precision); offloaded to CPU (§4.3).

### 4.3 Memory plan (single 16 GiB GPU)

ZeRO-2 + optimizer-state CPU offload (DeepSpeed config skeleton):

```yaml
# ds_config.json (relevant parts)
zero_optimization:
  stage: 2
  offload_optimizer: { device: cpu, pin_memory: true }
  contiguous_gradients: true
  overlap_comm: true
bf16: { enabled: true }
fp8: { enabled: true, fmt: "e4m3", scale_window: 1000 }
gradient_accumulation_steps: 16
gradient_clipping: 1.0
train_micro_batch_size_per_gpu: 1
activation_checkpointing:
  partition_activations: false
  contiguous_memory_optimization: true
```

- `micro_batch_size = 1`, `grad_accum = 16` → effective batch = 16 × 4096 = **65 536 tokens/step**.
- Activation checkpointing on every transformer block (24 stored ckpts × ~30 MiB activations = ~0.7 GiB).
- **Muon hidden weights only**: route `attn.q_a/q_b/o_lora_a/o_lora_b/expert.gate/up/down` through Muon optimizer; keep `embed_tokens, *_layernorm, router.gate, biases` on AdamW. Both optimizer states offloaded.

### 4.4 Sequence-length curriculum (mirrors DSv4)

| Phase | seq_len | tokens | attention | LR |
|---|---|---|---|---|
| 1 (warm-up) | 4 096 | 0–2 B | **dense** (override `compress_ratios` to all-zero, SWA only) | warmup 0 → peak (peak Muon=2e-2, AdamW=3e-4) |
| 2 (main dense) | 4 096 | 2 B–25 B | dense | cosine to 0.4× peak |
| 3 (sparse switch) | 8 192 | 25 B–35 B | enable CSA/HCA per `compress_ratios` (full V4 pattern) | cosine to 0.15× peak |
| 4 (long-context) | 16 384 | 35 B–40 B | sparse (CSA/HCA) | cosine to 0.10× peak |

We **do not** push to 64 K / 1 M at 16 GiB; the goal is to validate the curriculum, not deploy long-context. Final ckpt advertises `max_position_embeddings = 16 384` (still 4× the standard pretrain seq); full 65 K extension is deferred to a follow-up dedicated stage on rented A100/H100 if needed.

### 4.5 Step-time estimate

Reference: RTX 4090 BF16 dense @ ~165 TFLOPs sustained on a tuned 1-2 B model with TE FP8 amplification → ~200 TFLOPs effective. RTX 4070 Ti SUPER ≈ 0.65× of 4090 for memory-bound FP8 ≈ **130 TFLOPs effective**.

dsv4-mini per-step FLOPs at seq=4096, micro-batch=1, fwd+bwd ≈ `6 × activated_params × tokens = 6 × 0.33 B × 4096 = 8.1 TFLOPs/forward-pass × 3 (fwd+bwd+1 ckpt recompute) ≈ 24 TFLOPs/step`.

Step time ≈ `24 TFLOPs / 130 TFLOP/s ≈ 0.19 s` raw + ~0.3 s offload sync overhead = **~0.5 s/step**.

40 B tokens / 65 536 tokens/step = **610 k steps × 0.5 s ≈ 85 hours ≈ 3.5 days**.
Add overhead (eval, ckpt I/O, warm restart) → realistic wall-clock **4-7 days continuous**.

This fits ckl's stated 3-7 day budget.

### 4.6 Checkpointing

- Save every 5 B tokens (~10 hrs, 8 ckpts total). Each ckpt: ~6 GiB on disk (BF16 weights + optimizer).
- Use DeepSpeed `--save-checkpoint-tag` with rotating retention of last 3 + final.

---

## 5. Eval & acceptance

### 5.1 Pre-train metrics (live)

| Metric | Target | When |
|---|---|---|
| Val PPL on FineWeb-Edu held-out | ≤ 18 by 25 B tokens | every 1 B tokens |
| Val PPL on Chinese-Fineweb-Edu held-out | ≤ 30 by 25 B tokens | every 1 B tokens |
| Router entropy per MoE layer | ≥ 0.95 × log(32) (≈ 3.3) — reasonably uniform | every 100 M tokens |
| Per-expert utilisation σ/μ | ≤ 0.3 (sane balance) | every 500 M tokens |
| MTP head loss / main loss | ≤ 1.3× | every 100 M tokens |
| Loss spike count | < 5 | continuously, restart-from-ckpt rule if >3 spikes/24h |

### 5.2 Post-train (downstream)

Run `lm-evaluation-harness` (Python, fine to use; cold-path) at the final ckpt:

| Bench | Target | Honest expectation |
|---|---|---|
| C-Eval (zh) | > 30 (random 25) | "method works, not competitive" |
| MMLU (5-shot) | > 32 | same |
| HumanEval | > 8 (pass@1) | > 0 means MoE+CSA didn't break code |
| GSM8K (8-shot, CoT) | > 6 | same |
| BoolQ / ARC-easy | > 55 / > 40 | sanity; expected for ~330 M activated |

**Accept = method runs to completion without divergence + scores beat random baselines + all ablation toggles (mHC on/off, CSA on/off, MTP on/off) produce monotone PPL response.** We are validating the method on a small surface; we are not chasing leaderboards.

### 5.3 Ablation schedule (in-flight, single seed each)

After main run, do four 5 B-token re-runs from a 20 B ckpt:
1. Baseline (full dsv4-mini).
2. mHC off (`hc_mult=1`).
3. CSA/HCA off (override `compress_ratios` to zeros — pure SWA-128).
4. MTP off (`num_nextn_predict_layers=0`).

Each re-run ≈ 0.5 day. Drop on disk for `docs/experience/wins/`.

---

## 6. Risks & fallbacks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Muon on top of FP8/TE has open numerical issues at 1 B scale | M | major | Pre-flight: 2 B-token smoke run on subset, watch grad-norm. Fallback: AdamW-only (lose ~1.5× efficiency, plan still works in 5-9 days). |
| ZeRO-2 offload bandwidth bottlenecks step time (PCIe Gen4 x16 ~32 GB/s) | M | step → 1 s | Move some Muon momentum back to GPU if margin allows; consider 8-bit Adam state for the AdamW group. |
| MoE 32-expert routing collapses (1-2 dead experts on tiny scale) | M | quality cliff | Watch utilisation σ/μ; if any expert <2% load by 5 B tokens, re-init that expert from a healthy expert + warm restart. Standard DSv3 trick. |
| CSA Lightning Indexer kernel not in transformers HF impl yet | H | blocks training | Use the **NeMo-Automodel** path (it has the indexer); or fall back to dense attention for layers where `compress_ratios=4` (HCA-only run, still valid V4 ablation). |
| 16 GiB OOM during sparse-switch phase (extra indexer KV) | M | crash | Prepared: drop to `seq=2048` for phase 3 at the cost of weaker long-ctx behaviour, or enable CPU activation offload. |
| Disk fills (600 GB raw + 80 GB tokenised + ckpt) | L | crash | Pre-flight `df -h`; tokenise stream-by-stream and delete raw shards as they're consumed. |
| Tokenisation time blows the schedule | M | -1 to -2 days | Use the official DSv4 tokenizer instead of training our own (§3.1). |
| China data-source URLs change/auth-gate | L | re-source | All 6 slices have at least 2 mirrors (ModelScope + hf-mirror.com). |

---

## 7. Hand-off to ARLE runtime

**Goal**: take the trained `dsv4-mini` checkpoint and serve it through `infer/` Rust + CUDA. This is where ARLE eats the food we cooked.

### 7.1 Checkpoint conversion

PyTorch state_dict → safetensors using DSv4's published export script (NeMo-Automodel ships one). Tensor naming **does not match** today's `crates/deepseek-spec/`:

| dsv4-mini tensor (HF V4) | `crates/deepseek-spec/` today | Status |
|---|---|---|
| `model.layers.{L}.self_attn.q_a_proj.weight` (q-LoRA A) | covered (`q_a_proj`) | ✅ |
| `model.layers.{L}.self_attn.q_b_proj.weight` (q-LoRA B) | covered (`q_b_proj`) | ✅ |
| `model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight` | covered | **stale** — V4 has no kv-LoRA, this name is unused |
| `model.layers.{L}.self_attn.o_lora_a.weight` (o_groups × rank) | **missing** | needs new spec entry |
| `model.layers.{L}.self_attn.o_lora_b.weight` | **missing** | needs new spec entry |
| `model.layers.{L}.self_attn.k_proj.weight` (single-KV-head GQA, no LoRA) | **missing** | needs new spec entry |
| `model.layers.{L}.self_attn.v_proj.weight` | **missing** | needs new spec entry |
| `model.layers.{L}.self_attn.indexer.{w_q,w_k,w_o}.weight` | **missing** | needs new spec entry (Lightning Indexer) |
| `model.layers.{L}.self_attn.compressor.weight` | **missing** | needs new spec entry |
| `model.layers.{L}.hyper_connection.{P,Q}.weight` (mHC matrices) | **missing** | needs new spec entry |
| `model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight` | covered | ✅ |
| `model.layers.{L}.mlp.gate.weight` (router) | covered | ✅ |

**Action item (post-pretrain, separate commit):** extend `crates/deepseek-spec/src/lib.rs` with a `DeepSeekV4LayerTensorNames` variant that:
- replaces `DeepSeekMlaTensorNames` with `DeepSeekV4AttentionTensorNames` (q-LoRA + single-KV-head k/v + o-LoRA × o_groups + indexer + compressor),
- keeps `DeepSeekMoeTensorNames` unchanged,
- adds `DeepSeekHyperConnectionTensorNames` for mHC,
- adds new fields on `DeepSeekConfig`: `o_lora_rank, o_groups, index_n_heads, index_head_dim, index_topk, num_hash_layers, sliding_window, hc_mult, hc_sinkhorn_iters, compress_ratios: Vec<u32>, scoring_func: enum, swiglu_limit`.

This is the "V4 正式 delta" hook the substrate plan §0 / §9 tagged. Now we know exactly which fields and tensors.

### 7.2 Runtime kernel gap (CUDA backend, ARLE today)

`grep`-ed `infer/src/ops/` and `crates/cuda-kernels/csrc/`:

| Op needed by V4 | ARLE CUDA status |
|---|---|
| MLA prefill / decode | `mla_decode.cu` is a P0'' BF16 stub; substrate plan §6.1.1 has the FlashInfer plan. **Note: V4 does not need MLA at all.** This work can be deprioritised in favour of V4's actual attention path. |
| Single-KV-head GQA + q-LoRA + o-LoRA | not implemented; standard `prefill_attention.cu` covers GQA but not o-LoRA grouping. **Gap.** |
| CSA: stride-4 compressor + Lightning Indexer (top-k attention) | not implemented; Lightning Indexer is closest to FlashInfer's "page select" but still needs custom kernel. **Gap.** |
| HCA: stride-128 dense-on-compressed | implementable as standard dense attention on a small KV stream; reuse `prefill_attention.cu`. **Easy.** |
| Sliding-window-128 | not natively kernel-fused; today's prefill kernel does causal but not SWA mask. **Gap, easy.** |
| Sinkhorn-projected hyper-connection (4×4 matrix per layer) | trivial CPU+GPU mix; **trivial.** |
| MTP head | `model/deepseek/` has hooks but kernel path same as a normal layer. **Easy.** |
| MoE grouped-GEMM | substrate plan calls for cutlass-MoE; **Gap.** Critical for V4. |
| FP8 weights load + dynamic activation scaling | `decode_attention_varlen_fp8.cu` exists, but no end-to-end FP8 weight loader. **Gap.** |
| `sqrtsoftplus` router scoring | trivial fused kernel; **trivial.** |

**Follow-up work to actually serve dsv4-mini through ARLE** (separate plans, not this one):
1. `crates/deepseek-spec/`: V4 tensor-name + config extension (above).
2. `crates/cuda-kernels/csrc/`: cutlass grouped-MoE + Lightning Indexer + SWA-128 fused prefill.
3. `infer/src/model/deepseek/`: replace MLA path with V4 attention block; add mHC; add CSA/HCA layer dispatch via `compress_ratios`.

The pre-train side (this plan) is **independent** of those — we use stock PyTorch + transformers/NeMo for training, then hand off the safetensors blob.

### 7.3 Inference verification path (post-conversion)

Once V4 spec + kernels land:
1. Run dsv4-mini through transformers reference, snapshot top-32 logits per token on a 128-token prompt → `infer/test_data/dsv4_mini.json`.
2. `infer/tests/e2e_dsv4.rs` (new): load safetensors via ARLE, run same prompt, assert top-32 prob L1 ≤ 1e-3 (matches the existing Qwen3.5 acceptance bar).
3. `scripts/bench_guidellm.sh dsv4-mini-cuda-rtx4070tis` (per ARLE Benchmarks §): write win entry to `docs/experience/wins/`.

---

## 8. Open questions / [假设] needing confirmation

1. **`scoring_func: "sqrtsoftplus"` formula** — `sqrt(softplus(x))` is the natural reading; not confirmed against vLLM impl. Action: read `vllm/model_executor/models/deepseek_v4.py` once that lands.
2. **Hashed positions inside Lightning Indexer** (`num_hash_layers=3`) — exact hashing function (FNV? CRC? learned hash?) not in public docs. Action: same.
3. **Muon group split policy** — which params Muon vs AdamW. We mirror Moonlight conventions; DSv4 may differ. Action: scan DSv4-Pro Muon config in NeMo-Automodel's `dsv4-flash.md`.
4. **GRPO post-training** — DSv4 used GRPO + on-policy distillation. For dsv4-mini we ship pre-train only (post-training out of scope per "method ≠ score").
5. **`weight_block_size = [128, 128]` block-FP8** — TE supports this; need to confirm Ada Lovelace gets the speedup vs per-tensor scaling.

---

## 9. Definition of Done

- [ ] `dsv4-mini` config.json materialised at `models/dsv4-mini/config.json` matching §2 exactly.
- [ ] 49 152-vocab tokenizer trained (or DSv4 tokenizer adopted per fallback) and persisted at `models/dsv4-mini/tokenizer.json`.
- [ ] All 6 data slices downloaded, tokenised, and packed to `seq_len=4096`; total ≥ 40 B tokens, mix matches §3 ±2%.
- [ ] NeMo-Automodel (or fallback DeepSpeed) configured with this plan's hyper-params; pre-flight 2 B-token smoke runs without divergence.
- [ ] Full pre-train completes (~4-7 days); final ckpt at `models/dsv4-mini/checkpoints/final/`.
- [ ] Live metrics §5.1 satisfied at completion.
- [ ] Downstream evals §5.2 run; results table written to `docs/experience/wins/<date>-dsv4-mini-pretrain.md`.
- [ ] 4 ablations §5.3 complete; entries appended to the same wins file.
- [ ] Follow-up tickets opened for runtime hand-off §7 (spec extension, kernel work, e2e test) — **explicitly out of scope for this plan**.

---

## References

- DeepSeek V4 model card + config.json: `https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro` (released 2026-04-24)
- DeepSeek V4 Flash config.json: `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base/blob/main/config.json`
- HF blog "DeepSeek-V4: a million-token context that agents can actually use": `https://huggingface.co/blog/deepseekv4`
- DSv4 hybrid attention deep dive: `https://docs.bswen.com/blog/2026-04-25-deepseek-v4-1m-context-hybrid-attention/`
- DSv4 review: `https://artgor.medium.com/deepseek-v4-review-...`
- NeMo-Automodel V4-Flash guide: `https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/guides/llm/dsv4-flash.md`
- DeepSeek V3 Technical Report: `https://arxiv.org/abs/2412.19437`
- Muon Scalability: `https://arxiv.org/abs/2502.16982`
- Moonlight (Muon at scale): `https://arxiv.org/abs/2502.16982` (same paper)
- ARLE substrate plan: [`docs/plans/2026-05-05-deepseek-v4-small-substrate.md`](2026-05-05-deepseek-v4-small-substrate.md)
- ARLE deepseek-spec source: [`crates/deepseek-spec/src/lib.rs`](../../crates/deepseek-spec/src/lib.rs)
- China data mirrors: ModelScope (`https://modelscope.cn`), hf-mirror.com
