# DSv4 算子全景图 + Flash-默认接入计划（打基础）

> 目标：**FlashMLA 作为默认 attention 接入 → 把 GPU-native forward 基础打通 →
> 开始 A/B**。本文是 DSv4 所有算子的权威梳理，按 forward 顺序排列，
> **P 节点（prefill）/ D 节点（decode）/ P+D（共用）** 单独标注，并标注
> 每个算子的实现状态与 Flash-vs-legacy 归属。
>
> 证据来源（非推断）：`weights.rs` 的 `ffi::*` 调用点 +
> `crates/cuda-kernels/csrc/*` kernel 文件 + `reference.rs` 的
> `layer_forward`（CPU 正确参考序列，作为 correctness oracle）。

---

## 0. 方向与基础（先打好）

1. **Flash 作默认**：prefill 用 `arle_flashmla_sm90_sparse_prefill_fwd`，
   decode 用 `arle_flashmla_sm90_sparse_decode_fwd`。legacy
   `dsv4_hybrid_attention_cuda` / `dsv4_swa_attention_cuda` 只在 SM<90 或
   显式 env-OFF 时 fallback，parity 通过后删除。
2. **基础打通的唯一硬门**：**P→D KV 交接**。当前 serving prefill 是无状态
   batched（`compute_top_level_logits`, cache=None），不写 per-slot KV；
   incremental decode（cache=Some）读空 KV → garbage。这是 A/B 之前必须
   先修的地基（见 §3）。
3. **正确性 oracle**：`ARLE_DSV4_INFER_REAL_REFERENCE=1` 的 CPU 参考
   （`reference.rs`，完整正确 forward）。GPU-native 每个 tranche 落地都对它
   做逐层 parity，再 A/B 比 perf。
4. **A/B 协议**：同 binary、同 shell、同 prompt、两次 env 翻转、并排跑
   （CLAUDE.md 蒸馏教训）。先 correctness PASS 再 perf。

---

## 1. 单层 DSv4 forward 算子序列（reference.rs `layer_forward` ground truth）

DSv4-Flash 每层结构：**Hyper-Connection(MHC) → Attention → MHC-post →
MoE-FFN → MHC-combine**。每个 attention 层按 `compress_ratio` 分三种 mode：
SlidingWindow(SW) / CompressedSparse(CSA) / HybridCompressed(HCA)。

### 1A. Hyper-Connection（MHC，超连接）— **P+D 共用**

| # | 算子 | FFI symbol | csrc | 状态 | 节点 |
|---|------|-----------|------|------|------|
| 1 | MHC 参数生成 (sigmoid mix + softmax) | `dsv4_mhc_params_cuda` | (misc) | DONE | P+D |
| 2 | MHC pre-mix (展开 hc_mult 流) | `dsv4_mhc_pre_cuda` | (misc) | DONE | P+D |
| 3 | MHC head-pre | `dsv4_mhc_head_pre_cuda` | (misc) | DONE | P+D |
| 4 | MHC expand | `dsv4_mhc_expand_cuda` | (misc) | DONE | P+D |
| 5 | MHC post (residual combine) | `dsv4_mhc_post_cuda` | (misc) | DONE | P+D |

### 1B. Attention 前处理 — **P+D 共用**

| # | 算子 | FFI / op | csrc | 状态 | 节点 |
|---|------|----------|------|------|------|
| 6 | attn pre-norm (RMSNorm) | `ops::rms_norm_batch_into` | (ops) | DONE | P+D |
| 7 | Q 下投影 wq_a → q_norm → wq_b | `ops::gemm` + `rms_norm` | (ops) | DONE | P+D |
| 8 | KV 投影 wkv → kv_norm | `ops::gemm` + `rms_norm` | (ops) | DONE | P+D |
| 9 | Q/K partial-RoPE + 准备 | `dsv4_prepare_qk_cuda` / `dsv4_prepare_qk_fused_cuda` | attention | DONE | P+D |

### 1C. 压缩 / 稀疏选择（CSA/HCA 层，compress_ratio>0）— **P 写 / D 增量**

| # | 算子 | FFI symbol | csrc | 状态 | 节点 |
|---|------|-----------|------|------|------|
| 10 | 压缩器 (block 平均池化 K→compressed) | `dsv4_compressor_update_cuda` | attention | DONE | **P 全量 / D 增量** |
| 11 | CSA top-k 选择 (indexer 打分) | `dsv4_csa_select_cuda` | attention | DONE | P+D |
| 12 | FlashMLA CSA indices 构建 | `arle_flashmla_csa_build_indices` | (misc/decode build) | DONE | **P** |
| 13 | FlashMLA HCA indices 构建 | `arle_flashmla_hca_build_indices` | (misc) | DONE | **P** |
| 14 | FlashMLA decode indices 构建 (GPU, block-paged) | `arle_dsv4_flashmla_decode_build_indices_cuda` | dsv4_flashmla_decode_build_indices.cu | DONE | **D** |

### 1D. Attention 核心 — **Flash 默认；P / D 分开**

| # | 算子 | FFI symbol | csrc | 状态 | 节点 |
|---|------|-----------|------|------|------|
| 15 | **FlashMLA 稀疏 prefill** (Flash 默认) | `arle_flashmla_sm90_sparse_prefill_fwd` | vendor/flashmla | DONE (V2.4) | **P** |
| 16 | 统一 KV pool 打包 [SW\|k_prepared\|compressed] | `arle_flashmla_csa_pack_kv` | (shim) | DONE (已修 cache-less) | **P** |
| 17 | pad 行填充 | `arle_flashmla_fill_pad_rows` | (shim) | DONE | **P** |
| 18 | **FlashMLA 稀疏 decode** get_meta | `arle_flashmla_sm90_sparse_decode_get_meta` | vendor/flashmla | DONE (D-4) | **D** |
| 19 | **FlashMLA 稀疏 decode** sched_meta | `arle_flashmla_sm90_sparse_decode_sched_meta` | vendor/flashmla | DONE (D-4) | **D** |
| 20 | **FlashMLA 稀疏 decode** fwd | `arle_flashmla_sm90_sparse_decode_fwd` | vendor/flashmla | DONE (D-4) | **D** |
| 21 | FP8 KV 打包 (decode 单 token, strided) | `arle_dsv4_fp8_kv_pack_strided_cuda` | dsv4_fp8_kv_pack.cu | DONE (D-3') | **D** |
| 22 | SW window ring 更新 | `dsv4_update_window_cache_cuda` | attention | DONE | **D**（P bootstrap） |
| L | *(legacy)* hybrid attention | `dsv4_hybrid_attention_cuda` | fused_attention.cu | DONE→**待删** | P+D fallback |
| L | *(legacy)* SWA attention | `dsv4_swa_attention_cuda` | fused_attention.cu | DONE→**待删** | P fallback |

### 1E. Tensor-Parallel attention（TP>1）— **P+D 共用**

| # | 算子 | FFI symbol | 状态 | 节点 |
|---|------|-----------|------|------|
| 23 | Q AllGather repack | `dsv4_tp_q_repack_cuda` | DONE | P+D |
| 24 | 输出 slice (o_group) | `dsv4_tp_out_slice_cuda` | DONE | P+D |
| 25 | output projection o_proj | `ops::gemm` (per o_group) | DONE | P+D |
| 26 | attn all-reduce | `layer_communicator` (NCCL) | DONE | P+D |

### 1F. MoE-FFN — **P 全量 token / D 单 token；EP 分布**

| # | 算子 | FFI / op | csrc | 状态 | 节点 |
|---|------|----------|------|------|------|
| 27 | FFN pre-norm (RMSNorm) | `ops::rms_norm` | (ops) | DONE | P+D |
| 28 | 路由 gate (matvec + bias + sigmoid/softmax + top-k) | `dsv4_route_cuda` (GAP-B block-parallel) | dsv4_route.cu | DONE (GAP-B) | P+D |
| 29 | EP mask/indices by rank | `deepseek_mask_indices_by_ep` | deepseek_mask_indices_by_ep.cu | DONE | P+D（EP>1） |
| 30 | EP dispatch（native-DeepEP / NCCL 仿真） | DeepEP `Buffer::dispatch` | (deepep-sys) | **PARTIAL**（native B-3.3 落 fwd 部分，多 rank fallback NCCL） | P+D（EP>1） |
| 31 | routed expert SwiGLU (w1 gate→silu·clamp, w3 up, w2 down) | `dsv4_grouped_gemm` / `dsv4_deepgemm_ops` / `deepgemm_native` | dsv4_grouped_gemm.cu, dsv4_deepgemm_ops.cu, deepgemm_native.cu | DONE（weights-loaded 时） | P+D |
| 32 | 激活量化 (w4_fp8) | `w4_fp8_activation_quant` | w4_fp8_activation_quant.cu | DONE | P+D |
| 33 | 量化 GEMV (单 token decode) | `quantized_gemv` / `quantized_gemv_mma`(GAP-A) | quantized_gemv*.cu | DONE / GAP-A Phase4 待接 | **D** |
| 34 | EP combine | DeepEP `Buffer::combine` | (deepep-sys) | PARTIAL（同 30） | P+D（EP>1） |
| 35 | shared expert SwiGLU | `add_shared_expert` | mlp.rs | DONE | P+D |
| 36 | MoE all-reduce (EP) | `post_moe_expert_all_reduce` | (NCCL) | DONE | P+D |
| 37 | MHC combine (residual) | `dsv4_mhc_post_cuda` | (misc) | DONE | P+D |

### 1G. 顶层（整个序列一次）— **P 出首 token logits / D 出每步 logits**

| # | 算子 | FFI / op | 状态 | 节点 |
|---|------|----------|------|------|
| 38 | embedding lookup | (host/gather) | DONE | P |
| 39 | 初始 HC stream 构建 | `initial_hc_stream_from_embeddings` | DONE | P |
| 40 | 最终 head norm (head_hc) | `head_hidden_from_stream` | DONE | P+D |
| 41 | lm_head → logits | `common::compute_logits_batch` | DONE | P+D |
| 42 | logits bf16→f32 | `arle_bf16_to_f32_cuda` | DONE | P+D |

---

## 2. P 节点 vs D 节点 算子归属总表

> PD-disaggregation 视角：P 节点 compute-bound（处理整 prompt），
> D 节点 memory-bound（逐 token）。**共用算子**两边都要正确。

### 仅 P 节点（prefill-only）
- FlashMLA 稀疏 prefill fwd (#15) + 统一 KV pool 打包 (#16) + pad 填充 (#17)
- CSA/HCA indices 构建 (#12, #13)
- 压缩器**全量**构建 (#10) — 把整 prompt 的 K 池化成 compressed
- embedding / 初始 HC stream (#38, #39)
- **KV cache WRITE**（关键）：prefill 必须把 SW window ring + compressed +
  FP8 pool **写满**，供 D 节点读。← **当前缺失，见 §3**

### 仅 D 节点（decode-only）
- FlashMLA 稀疏 decode get_meta/sched_meta/fwd (#18, #19, #20)
- decode indices 构建 (#14)
- FP8 KV 单 token 打包 strided (#21)
- SW window ring 增量更新 (#22)
- 压缩器**增量**更新 (#10 增量分支)
- 量化 GEMV 单 token (#33, GAP-A)

### P+D 共用（两节点都跑，必须都对）
- 全部 MHC 超连接 (#1–5, #37)
- attn 前处理 RMSNorm + Q/KV 投影 + RoPE prep (#6–9)
- CSA top-k 选择 (#11)
- TP AllGather/slice/all-reduce + o_proj (#23–26)
- MoE：gate 路由 (#28)、EP dispatch/combine (#30,#34)、routed/shared experts
  (#31,#35)、量化 (#32)、all-reduce (#36)
- 顶层 head norm / lm_head / logits 转换 (#40–42)

---

## 3. 地基硬门：P→D KV 交接（A/B 之前必修）

**现状（证据）**：
- serving prefill = `compute_gpu_logits_after_prefill` → `compute_top_level_logits`
  （无状态 batched，cache=None，**不写 per-slot KV**）。
- serving decode = `compute_gpu_logits_after_decode` →（`ARLE_DSV4_INCREMENTAL_KV=1`
  时）`compute_top_level_logits_incremental` →
  `forward_transformer_layer_stream_incremental_into`（cache=Some，
  **读 per-slot KV**）。
- 中间**没有**把 prefill 的 KV 写进 `state.incremental` / `cache.attention`
  的环节 → decode 读空 KV → garbage（"4062 0.0000…"）。

**修复方向（待 Plan 细化）**：prefill 走能写 KV 的路径（incremental
emit_logits=false 预热，或 batched-prefill 后做一次 KV bootstrap），让
D 节点首步读到 prefill 写好的 SW window + compressed + FP8 pool。
FlashMLA decode 的 SW bootstrap hook（`dsv4_flashmla_sw_bootstrap_hook`）
已为此预留——但它从 `cache.window_gpu` 取数，而 cache-less prefill 没填它。

**验证**：修好后用 reference oracle 比对，prompt "137+269" → "406"。

---

## 4. 实现状态汇总

| 状态 | 算子 |
|------|------|
| **DONE** | MHC 全套、attn 前处理、RoPE prep、压缩器、CSA select、FlashMLA prefill(V2.4)、FlashMLA decode(D-4)、FP8 KV pack、SW window、TP attention、gate 路由(GAP-B)、routed/shared experts、量化、顶层 |
| **PARTIAL** | native-DeepEP dispatch/combine（多 rank 仍 fallback NCCL，B-3.3）；GAP-A MMA GEMV（kernel 落地，Rust dispatch 待接 Phase4） |
| **STUBBED/缺失** | **P→D KV 交接**（地基硬门）；GPU 路径默认 `GPU_FULL_LAYERS=0` + `LOAD_LAYER_WEIGHTS=0` 不跑真实层 |
| **待删（parity 后）** | legacy `dsv4_hybrid_attention_cuda` / `dsv4_swa_attention_cuda` |

---

## 5. 下一步顺序（打基础 → A/B）

1. **拿 reference oracle**：`INFER_REAL_REFERENCE=1` 跑出 "406"，锁定正确基线。
2. **修 P→D KV 交接**（§3）——地基硬门。
3. **逐层 parity**：GPU-native（GPU_FULL_LAYERS=全部 + LOAD_LAYER_WEIGHTS=1 +
   Flash 默认）对 reference，定位首个发散层，逐 tranche 修到 bit-parity。
4. **Flash 默认翻转**：parity PASS 后 `ARLE_DSV4_FLASHMLA_{PREFILL,DECODE}`
   默认 ON，删 legacy。
5. **开始 A/B**：correctness PASS 后比 perf（TTFT/TPOT/throughput），
   P 节点 / D 节点分别测。
