# Multimodal (Vision) — ARLE Cross-Backend Plan

Status: PROPOSED · Owners: ckl + claude · Date: 2026-05-01
Targets: Qwen2.5-VL (7B/3B), Qwen3-VL (8B/2B); CUDA + Metal; image first, video phase B.
Ground rules: 集众家之所长 → vLLM v1 调度，SGLang RadixCache 哈希注入，mlx-vlm 接口边界，自研 Metal-first ViT。

每一节都按 "唯一非平凡决策 + 可验证退出条件" 写。已知能从 HF reference 推出的常识不再重复（参见 §2 表格）。

---

## 0. Decision Pack（不可妥协的 6 个决策）

| # | 决策 | 理由（一句话） | 拒绝的备选 |
|---|---|---|---|
| D1 | **单进程 Rust runtime**，ViT 与 LLM 同图、同 backend trait、同调度循环 | 拒绝 vLLM EPD 与 llama.cpp mmproj 的进程/文件分割 — Rust 无 GIL，IPC 在我们这是负优化 | EPD、mmproj 双 GGUF |
| D2 | 调度器抽 `EncoderCacheManager`（compute_budget + cache_slots 双预算），prefill 跨 mm 边界时 **rollback to start_pos** | vLLM v1 已经验证：少了这个，多图请求会饿死纯文本流 | TGI 无特殊调度（QPS 抖动） |
| D3 | **两层正交缓存**：(a) `RadixCache` 用 `mm_hash → pad_value` 注入占位 token，命中"同前缀 + 同图像"省 LLM 前向；(b) `EncoderCache(LRU<mm_hash → vision_embeds>)` 命中"任意上下文 + 同图像"省 ViT 前向。两者命中场景不重叠 | SGLang 用 (a) 是优雅；vLLM 用 (b) 是必需；ARLE 都做 — (a) 复用既有 KV-tier，(b) 只活在 GPU 内存 | 任一只做一层 |
| D4 | `Backend` trait 增加 **`encode_vision`** 与 **`resolve_input_embeds`** 两个新接口：前者跑 ViT，后者把含 pad_value 的 input_ids 解析成 hidden_states（embed_table 查表 + vision_embeds scatter 一体） | 切口窄；让 CUDA/Metal 各自决策内部 kernel；resolve 单独存在以避免 pad_value 漏到 sampler/lm_head | 给每个 backend 加一套并行 op 表；或在 LLM forward 里 inline 解析 |
| D5 | Metal ViT 注意力 = 复用 `infer/src/backend/metal/AGENTS.md` §7 的 packed-varlen + additive-mask + per-row offset shader（**不**写每窗 SDPA Python loop） | 目标 ≥ 2× mlx-vlm（per-window Python loop）— §11.6 上线门槛；shader 已存量 | mlx-vlm 的 per-window `scaled_dot_product_attention` 循环 |
| D6 | 图像预处理在 HTTP handler 同进程同步执行（rayon fan-out），结果以 `MmFeature { pixels, grid_thw, mm_hash, mrope_positions }` 进调度器 | Rust 无 GIL → SGLang 的 ProcessPool 是不必要的复杂度；vLLM 的 SHM 也是 | 任何形式的子进程 / SHM / IPC |

---

## 1. 架构图

### 1.1 顶层数据流（请求 → 字节）

```
                  HTTP / OpenAI /v1/chat/completions
                                │  content:[{type:"image_url", url|data|uuid}]
                                ▼
   ┌──────────────────── infer/src/http_server/openai_v1.rs ────────────────────┐
   │  parse_content_parts() ─► ImageInput{bytes,mime,uuid?}                     │
   │      │                                                                     │
   │      ▼   rayon::scope                                                       │
   │  vision::ingest::preprocess(ImageInput) ─► MmFeature                        │
   │      ├─ image crate: decode (jpeg/png/webp)                                 │
   │      ├─ smart_resize(factor)              ← model-specific factor (28/32)   │
   │      ├─ normalize(mean,std)               ← model-specific stats            │
   │      ├─ pack_patches → pixel_values [N_patch, C·T·P·P]                      │
   │      ├─ blake3(canonical) → mm_hash                                         │
   │      └─ derive pad_value = u32(mm_hash[0..4]) | 0x8000_0000                 │
   │                                                                             │
   │  tokenizer::expand_vision_markers(text, mm_features)                        │
   │      └─ <|vision_start|> + (pad_value)×N_llm + <|vision_end|>               │
   │                                                                             │
   │  mrope::compute_positions(input_ids, mm_features) → [3, S]                  │
   └──────────────┬─────────────────────────────────────────────────────────────┘
                  ▼
   IncomingRequest { tokens, mm_features:Vec<MmFeature>, mrope_pos }
                  │
                  ▼
   ┌────────────── infer/src/scheduler/{cuda|metal}/ ─────────────────────────────┐
   │   admission ── encoder_budget? cache_slots? ──► chunked prefill plan          │
   │           rollback-to-start_pos when mm cannot fit                            │
   │                                                                               │
   │   per tick:                                                                   │
   │     (1) encode_vision pass  ── EncoderCacheManager hit ► skip                 │
   │                              ── miss ► backend.encode_vision(..)              │
   │     (2) text prefill / decode pass                                            │
   │     (3) DeepStack injection at decoder layers 0/1/2 (Q3-VL only)              │
   └──────────────┬─────────────────────────────────────────────────────────────────┘
                  ▼
   LoadedInferenceEngine ── CUDA   ► crates/cuda-kernels (FlashInfer non-causal +
                                     conv3d + 2D-RoPE + ln-bias + gelu_tanh)
                       \\── Metal  ► crates/mlx-sys (varlen-additive-mask SDPA +
                                     conv3d + 2D-RoPE + ln-bias + gelu_tanh)
```

### 1.2 调度器 tick 内的多模态资源管理（vLLM v1 模式）

```
   每 tick 起点
        │
        ▼
   ┌─────────── schedule_running ───────────┐
   │ for req in running:                     │
   │   for mm in req.mm_features:            │
   │     if mm.range ∩ chunk_range:          │
   │        if cached(mm.hash): use_cached() │
   │        elif compute_budget>=mm.cost     │
   │           and cache_slots>=mm.tokens:   │
   │             schedule_encode(mm)         │
   │             compute_budget -= mm.cost   │
   │             cache_slots    -= mm.tokens │
   │        else:                            │
   │             num_new_tokens =            │
   │                start_pos − num_computed │  ← 关键：rollback
   │             break                       │
   └────────────────────┬────────────────────┘
                        ▼
   ┌─────────── encoder pass ────────────┐    ┌──────── llm pass ────────┐
   │ batch ALL scheduled mm in one ViT   │ →  │ prefill / decode as 1D    │
   │ forward (concat pixel_values along  │    │ token stream; lookup      │
   │ N_patch axis, cu_seqlens segments)  │    │ embed_table[pad_value]    │
   │ → gather slice into encoder_cache   │    │ overridden by             │
   │   keyed by mm_hash                  │    │ vision_embeds[mm_hash]    │
   └─────────────────────────────────────┘    └──────────────────────────┘
                                                       │
                                            (Q3-VL only) ▼
                                       ┌─────────────────────────────┐
                                       │ deepstack_embeds[0..2] +=   │
                                       │ hidden_states at layers 0,1,2│
                                       │ at visual_pos_mask positions │
                                       └─────────────────────────────┘
```

### 1.3 两层正交缓存（D3 详解）

```
   两层缓存命中场景不重叠：

   Layer A — RadixCache (KV pages, T0/T1/T2/T3, 复用既有 prefix_cache.rs)
     key  = blake3(tokens_le || mm_hash_of_run)        # 见 §4.5
     val  = paged KV blocks
     hit  ⇔ (前缀 token 一致) ∧ (前缀里所有图像 hash 一致)
     省   = LLM prefill（包含 vision token 区间的 KV）

   Layer B — EncoderCache (LRU<mm_hash → vision_embeds>, GPU-only, 不持久化)
     key  = mm_hash
     val  = vision_embeds [N_llm, out_hidden]  +  deepstack_embeds[3]
     hit  ⇔ 同图像出现在任意上下文（包括前缀不同/拼接顺序不同）
     省   = ViT forward（不省 LLM）

   场景矩阵（hit_A, hit_B）→ 路径
     (1, 1)   都命中     → skip ViT, skip LLM-prefill       # 同图同前缀（多轮对话最常见）
     (1, 0)   仅 A       → skip ViT, skip LLM-prefill       # A 持久化在 T1/T2 KV-tier；
                                                              B 已 LRU 驱逐。KV 是 ViT 的下游
                                                              产物，A 命中 ⇒ ViT 输出已被 KV
                                                              吸收，不需要重跑（仅当后续轮次
                                                              需要原始 vision_embeds 才回填 B —
                                                              图像默认不需要，DeepStack 才需要，
                                                              见下方注解）
     (0, 1)   仅 B       → skip ViT, run  LLM-prefill       # 同图新前缀
     (0, 0)   全 miss    → run ViT, run LLM-prefill, fill 双层缓存

   两层都用 mm_hash 作键根 — 统一了图像身份语义，不会互相污染。

   注解 — DeepStack 例外（Q3-VL）：deepstack_embeds 不进 KV cache（它是中间残差注入，
   不是 attention KV）。所以 Q3-VL 的 (1, 0) 场景必须 *回退* 到 (0, 0)：
   即使 A 命中也要重跑 ViT 才能拿到 deepstack_embeds 用于后续 decode 步。
   纯 Q2.5-VL 路径无此例外。
```

### 1.4 双后端 ViT 执行图（同一前端 → 两条 kernel 路径）

```
                  pixel_values [N_patch, 1176|1536]   grid_thw [G,3]
                              │                              │
                              ▼                              │
      ┌───────────── conv3d patch_embed ─────────────┐       │
      │  CUDA: cuDNN cudnnConvolutionForward          │       │
      │  Metal: mlx::conv_general_dilated(spatial=2D, │       │
      │         temporal=1) — fold T into batch       │       │
      └────────────────┬─────────────────────────────┘       │
                       ▼  hidden [N_patch, 1280|1152]        │
              window_index permute  ◄────  build cu_seqlens / cu_window_seqlens
                       │                       (Q2.5-VL: 2 sets, full+window)
                       ▼                       (Q3-VL:   1 set,  full only)
   ┌─── for layer in 0..depth ───────────────────────────────────────┐
   │  norm1                                                           │
   │   ├─ Q2.5-VL: RMSNorm                                             │
   │   └─ Q3-VL:   LayerNorm(γ,β)                                      │
   │  attn(qkv,proj):                                                  │
   │   ├─ CUDA: FlashInfer non-causal varlen, cu_seqlens 选择切换       │
   │   └─ Metal: packed-varlen + additive mask（§7 模式），             │
   │             同一 shader 既跑 full 又跑 window（mask 形状不同）      │
   │   2D-RoPE：head_dim/2 分裂为 H, W；分别跑 1D-RoPE 后 concat        │
   │  norm2 + mlp                                                      │
   │   ├─ Q2.5-VL: SwiGLU(SiLU)                                        │
   │   └─ Q3-VL:   linear_fc1 → gelu_pytorch_tanh → linear_fc2         │
   │  (Q3-VL: layer ∈ {8,16,24}) → deepstack_merger_list[i](h) → out_i │
   └────────────────┬─────────────────────────────────────────────────┘
                    ▼  reverse_indices(window_index permute)
              merger: norm + view(N/4, 4·d) + mlp → out_hidden_size
                    ▼
              vision_embeds [N_llm, 3584|4096]
              deepstack_embeds [3][N_llm, 4096]   (Q3-VL only)
```

---

## 2. 收敛过的关键参数表（剩下都从 HF config 派生，不复述）

| 参数 | Q2.5-VL-7B | Q3-VL-8B | 影响 |
|---|---|---|---|
| ViT depth × hidden | 32 × 1280 | 27 × 1152 | 编码 FLOPs 决定 `encoder_compute_budget` 单位 |
| patch_size · spatial_merge_size · factor | 14 · 2 · **28** | 16 · 2 · **32** | smart_resize 量化步长 |
| temporal_patch_size | 2 | 2 | 静态图复制成 2 帧 |
| pixel_values 行宽 | 3·2·14·14 = **1176** | 3·2·16·16 = **1536** | 内存带宽 |
| ViT norm | RMSNorm | LayerNorm(bias) | Metal 必须实现 LN with bias |
| ViT MLP 激活 | SwiGLU/SiLU | gelu_pytorch_tanh | tanh-approx 必须严格匹配 |
| 注意力布局 | full + window(112)，full at {7,15,23,31} | 全 full | Q2.5-VL 需要双 cu_seqlens |
| 2D-RoPE θ | 10000 | 10000 + 学习的 2304 行 abs-pos | Q3-VL 多一个 bilinear interp 缓存 |
| merger 形状 | RMSNorm + 5120→5120 (GELU 严格) + 5120→3584 | LayerNorm + 4608→4608 (GELU tanh) + 4608→4096 | merger 是普通 MLP，复用 LLM linear |
| DeepStack 层 | 无 | 8/16/24 → 注入 LLM 层 0/1/2 | Q3-VL 路径多 3 个 merger 实例 |
| M-RoPE 模式 | 连续 [16,24,24] | **interleaved** [24,20,20] | RoPE kernel 加一个 axis-lookup 表参数 |
| 文本 rope_theta | 1e6 | 5e6 | 配置而已 |
| Image mean/std | OpenAI-CLIP | **[0.5,0.5,0.5]** | 必须按 model 绑定，不按 processor |
| 特殊 token | 151652 vs_start, 151655 image_pad, 151653 vs_end | 同上 | 同 |
| 默认 max_pixels | 12.85M | 16.78M | 服务端默认应进一步收紧到 ~1.6M |

每张图的 LLM 占位 token 数：`N_llm = grid_t · (grid_h/2) · (grid_w/2) = N_vit / 4`。
每张图的 mrope `current_pos` 推进量：`max(grid_h, grid_w)/2`（**不是** N_llm）。

---

## 3. Workspace Diff（按 §architecture 准入规则做最小切口）

### 3.1 新增模块（全部落在 `infer/`，不开新 crate；遵守 §architecture "无具体二消费者不切" 治理）

```
infer/src/vision/                                      ← 新模块（mod = vision）
   mod.rs              # 导出 + feature flag
   image.rs            # ImageInput, decode, smart_resize, normalize, pack_patches
   hash.rs             # blake3 mm_hash + EXIF-ImageID shortcut + pad_value derive
   mrope.rs            # 3D position_ids 计算（接收 token_type_ids + grid_thw）
   processor.rs        # rayon fan-out 入口；MmFeature 输出
   encoder_cache.rs    # LRU<mm_hash → DeviceTensor>，compute/slots 预算
   model/qwen25vl.rs   # ViT.forward + merger，用 backend trait
   model/qwen3vl.rs    # 同上 + DeepStack
   weights.rs          # safetensors → ViT 张量解析（visual.* 前缀）

infer/src/model/qwen25_vl.rs                           ← 新（语言端 Q2.5-VL；与 qwen3.rs 几乎一致，仅在 forward 入口加 resolve_input_embeds 调用）
infer/src/model/qwen3_vl.rs                            ← 新（语言端 Q3-VL；必须分叉，因为 DeepStack 在 layer 0/1/2 后注入，纯文本请求也走这条 hook 路径，hook 只是 no-op；不能复用 qwen35.rs 原版）

crates/mlx-sys/src/                                    ← 增量 FFI
   conv3d.rs           # mlx_conv_general_dilated 包装（spatial=2D + temporal=1）
   layernorm.rs        # mlx_fast_layer_norm 包装（mlx 已有）
   gelu.rs             # gelu / gelu_tanh 用 mlx unaries 组合包装
crates/mlx-sys/cpp/    # 不新增 .metal 文件；varlen-additive-mask SDPA 复用 §7 已有 shader

crates/cuda-kernels/csrc/
   misc/layernorm_bias.cu    # ~80 行；同时给 LLM 复用（备）
   misc/gelu.cu              # exact GELU + tanh-approx 两个变体
   attention/flashinfer_noncausal_varlen.cu  # FlashInfer 模板特化（causal=false）
   attention/conv3d_patch_embed.cu           # 走 cuDNN（推荐）或 im2col+gemm（fallback）
crates/cuda-kernels/build.rs                            ← 新增 -lcudnn 链接（feature 门禁）
```

### 3.2 现有文件的 surgical 改动（行号以本次调研结果为锚）

| 文件 | 行 | 改动 |
|---|---|---|
| `infer/src/server_engine/types.rs` | 6-24 | `CompletionRequest` 加 `mm_features: Vec<MmFeature>` |
| `infer/src/scheduler/types.rs` | 416-441 | `IncomingRequest` 同上 + `mrope_positions: Option<Mrope3>` |
| `infer/src/model.rs` | 28-45, 163-175 | `PrefillBatchRequest` / `MixedBatchRequest` 携带 `Option<&MmEmbeddingTable>` 与 `mrope_positions`；`GenerationState` 接 `deepstack_streams` |
| `infer/src/http_server/openai_v1.rs` | 205-237 | 删 `validate_text_only_content`；接入 `vision::processor::ingest()` |
| `crates/chat/src/lib.rs` | 59-79 | `OpenAiChatContent::Parts` 提供 `extract_images() → Vec<ImageInput>` |
| `infer/src/tokenizer.rs` | 34-52 | 新方法 `encode_with_vision(text, mm_features)` 内部插占位 token 序列 |
| `infer/src/model_registry.rs` | 137 | 增加 `Qwen2_5_VLForConditionalGeneration` / `Qwen3VLForConditionalGeneration` → 路由到 `qwen25_vl` / `qwen3_vl` 装载器 |
| `infer/src/weight_loader.rs` | 25-60 | 多前缀过滤：`visual.*` → vision shard，`model.language_model.*` → LLM shard（Q3-VL 嵌套） |
| `infer/src/prefix_cache.rs` | 114-140 | `RadixNode` 加 `mm_hash: Option<[u8;16]>`；fingerprint 计算改为 `blake3(tokens || mm_hash_of_run)` |
| `infer/src/backend.rs` | 61-85 | 新增独立 trait `VisionBackend: InferenceBackend`（含 `encode_vision` + `resolve_input_embeds`）；CudaBackend/MetalBackend 实现，CpuBackend 不实现；`LoadedInferenceEngine::as_vision() -> Option<&dyn VisionBackend>` 给 model_registry 在加载期判断 — VL 模型 + 无 vision 后端 = `LoadError::VisionRequiresGpuBackend`。InferenceBackend 不动，所有现有 mock/测试零修改。 |
| `infer/src/backend/cuda/bootstrap.rs` | 152-198 | 加载 vision shard、构造 `VisionTowerCuda`，挂到 engine |
| `infer/src/backend/metal.rs` | 135-150 | 同上但走 `VisionTowerMetal` |
| `infer/src/scheduler/cuda/core.rs` | (encoder admission 段) | 接入 `EncoderCacheManager`；prefill 计算时计入 mm 区间 |
| `infer/src/backend/metal/scheduler.rs` | (同 CUDA 对应位置) | 同上 |
| `crates/qwen35-spec/`、`crates/qwen3-spec/` | tensor 名表 | 追加 `visual.*` 与 `language_model.*` 前缀映射；版本字段 `vision_layout: Option<...>` |
| `Cargo.toml` (workspace) | deps | 加 `image = "0.25"`（PNG/JPEG/WebP），`fast_image_resize = "5"`（SIMD bicubic），`blake3 = "1"`（已有则只加 features），`reqwest = { features=["rustls-tls"] }`（如未拉） |

> 行 / 名 约定与现状一致；具体 line numbers 在实施 PR 内 freeze。

---

## 4. 模块详细设计（按合理实施顺序）

### 4.1 `vision::image` — CPU 预处理

输入 `ImageInput { bytes, mime, uuid? }`：

1. `image::load_from_memory_with_format(...)` → `DynamicImage`；任意输入色域统一 `into_rgb8()`（grayscale/CMYK/alpha 都先合成到 RGB 白底），与 HF `image.convert("RGB")` 行为一致。
2. EXIF orientation：`kamadak-exif` 读取 tag `0x0112`，按 8 种方向旋转。**EXIF ImageUniqueID（tag `0xA420`）只作为 cache lookup hint**（见 §4.2），**绝不**短路像素哈希 — 否则任何拷贝/伪造的 EXIF ID 都能让两张不同图像在缓存里别名。
3. `smart_resize(h,w,factor,min_pixels,max_pixels)` — 按 §2 表选 factor；`fast_image_resize` 走 bicubic（**fp32 内核**，AVX2/NEON 自动启用），输出仍 `RgbImage<u8>`。
4. mean/std 归一化：u8 → fp32 → 减均值除方差（SIMD 单 pass，乘加形式）；**最后阶段 cast 到 bf16** 输出，避免 ViT 入口再转。
5. `pack_patches`：reshape + permute（§Q2.5-VL §2.3 公式）→ `[N_patch, row_width]`，row_width 按模型表。**单图复制成 T=2 两份**（避免在 ViT 写两条 Conv 路径）。

**单请求内存预算（强制）**：累计 `Σ decoded_pixel_bytes ≤ 256 MiB`（默认；CLI/服务端可调）；超出的请求在 decode 阶段就 413。**单进程总预算**：preprocessing 工作集 ≤ `min(8 GiB, mem/4)`，溢出走背压（HTTP 503 + Retry-After）。

退出条件：`pixel_values.shape == (Σ N_patch, row_width)`，`grid_thw.shape == (num_images, 3)`，且能用 HF transformers 的 image_processor 在 1e-6 max-abs 误差内复现。

**异步契约**：在 axum handler 里走 `tokio::task::spawn_blocking(|| rayon::scope(...))`，rayon 池上限 = `num_cpus / 2`，handler 端 `await` 不阻塞 reactor。

### 4.2 `vision::hash` — 缓存键

```
mm_hash = blake3(
    model_id_u32_le             # 防止跨模型撞键
    || tenant_id_u64_le         # 多租户隔离；单租户填 0
    || row_width_u16_le         # 防止 14↔16 撞键
    || mean_std_u32x6_quantized # 不同归一化 → 不同键
    || canonical_pixels_bytes   # bf16 字节，按 Σ N_patch · row_width 顺序，规范字节序
)[..16]
```

**UUID / EXIF ImageUniqueID 仅作 lookup hint，不替代哈希**：

```
fn lookup_or_compute(input: ImageInput) -> MmHash {
    if let Some(hint) = input.uuid_or_exif_id() {            // 客户端 UUID 优先；其次 EXIF 0xA420
        if let Some((cached_hash, cached_canonical)) = HINT_CACHE.get(&hint) {
            // 必须二次校验：解码 + canonical 之后字节比对
            let canonical = decode_and_canonicalize(input.bytes);
            if canonical_bytes_eq(&canonical, &cached_canonical) {
                return cached_hash;                          // 节省的是 blake3，不是验证
            }
            // 校验失败：客户端伪造 / 撞 hint，按真实哈希走
        }
    }
    let hash = blake3_full(input);
    HINT_CACHE.insert(input.hint, (hash, canonical_bytes));
    hash
}
```

这条规则非可选 — 跳过它就等于把缓存隔离权交给客户端。

```
pad_value = 0x8000_0000 | u32::from_le_bytes(mm_hash[0..4])
```

`pad_value` 顶位 1 → 与 tokenizer vocab id 严格不相交。但**整条 forward 路径都要把 pad_value 当哨兵处理**：见 §4.6 解析层。

### 4.3 `vision::mrope` — 一次性算 3D position

输入：扩展后的 `input_ids`（含 pad_value 序列）+ `grid_thw[]`。

伪码：

```
pos = (0,0,0)             # current scalar 三轴一致
for run in segment_by_token_type(input_ids, mm_hashes_to_grid):
    if run.is_text:
        emit (pos.0, pos.0, pos.0) for k in 0..len(run); pos += (len, len, len)
    elif run.is_image(grid):
        for h in 0..grid.h/2:
            for w in 0..grid.w/2:
                emit (pos.0, pos.1+h, pos.2+w)        # Q2.5-VL: T=0, H, W
        delta = max(grid.h, grid.w) / 2
        pos = (pos.0+delta, pos.1+delta, pos.2+delta)
    elif run.is_video(grid, sec_per_grid):
        ...     # Phase B
```

输出 `mrope_positions: [3, S]`（u32）。Q3-VL 的 interleaved 在 RoPE kernel 端用一张 64-长度的 `axis_lut: [u8; 64]` 表把每个 RoPE dim 映射到 `{0=T,1=H,2=W}`，**不在这里展开**。

### 4.4 `vision::encoder_cache` — 双预算

```
struct EncoderCacheManager {
    capacity_tokens: usize,       // 比如 32_768 vision-tokens
    compute_budget_per_step: u64, // 比如 2 * GFLOPs_per_step；动态 EWMA
    cached: HashMap<MmHash, Arc<DeviceVec>>,
    refcnt: HashMap<MmHash, u32>,
    lru:    IndexMap<MmHash, Instant>,    // freeable = refcnt == 0
}
```

调度器每 tick：
1. 收集本 tick 所有候选 mm，按 `(req_priority, mm.start_pos)` 排序。
2. 累加预算：
   - `compute_cost(mm) ≈ depth · (16·d²·N_patch + 4·d·N_patch²)`（FFN+attn 两项；attention 项在 N_patch≳hidden 时显著，max_pixels 边界必须计入）。
   - `slot_cost(mm) = N_llm`。
   - EWMA 用 ViT pass 实测 FLOPs/μs 校准 `compute_budget_per_step` 的"单位价格"，每 tick 修正。
3. 任一项超预算 → 当前 mm **不**调度，发起 rollback：把宿主 request 的 `num_new_tokens` 截到 `start_pos − num_computed`（vLLM v1 规则）。
4. 命中 cached → 仅给 `refcnt += 1`，不参与 compute_budget。

**编码批量化**：所有同 tick 调度的 mm 共享一次 `backend.encode_vision()`，pixel_values 在 N_patch 维 cat 起来，`cu_seqlens` 在 vision_tower 内部分段 — 这是 vLLM/SGLang 都没做透的点（他们多按一图一次）。我们一开始就批，因为 Metal 启动开销高、CUDA graph 喜欢一致性。

### 4.5 RadixCache 注入 — `prefix_cache.rs`

`pad_value`（高位 1）会被自然纳入 token-id 序列。RadixNode 增加 `mm_hash: Option<[u8;16]>`，fingerprint 改为：

```
fingerprint = blake3(
    tokens_le_bytes
    || (mm_hash if any(token & 0x8000_0000) else b"")
)
```

关键不变量：**同图同前缀 → 同 fingerprint**；**异图同前缀 → fingerprint 分叉**。这样 KV-tier coordinator 不需要改动，T1/T2 demote/promote 自动跟随。

**已知陷阱**：当一个 RadixNode 横跨「文本+vision_pad+文本」时，`mm_hash_of_run` 取该节点内**所有 vision run 的 mm_hash 串接 blake3**，避免节点切分粒度变得不稳定。节点 split 仍按 token 长度对齐（block size 内），不破坏 paged KV。

### 4.6 Backend trait 扩展（D4）— 独立 `VisionBackend` 子 trait

为避免 `CpuBackend` 与所有现有测试 mock 都被迫加新方法（哪怕 stub），新增能力**不**塞进 `InferenceBackend`，而是挂一个**可选的子 trait**：

```rust
// 既有 trait 不变
pub trait InferenceBackend { /* ... 现有定义保持 ... */ }

// 新增独立 trait — 只有 GPU 后端实现
pub trait VisionBackend: InferenceBackend {
    /// 跑 ViT
    fn encode_vision(
        &self,
        pixel_values:      &BfTensor,             // [Σ N_patch, row_width]
        cu_seqlens_full:   &[u32],
        cu_seqlens_window: Option<&[u32]>,        // Q2.5-VL 才有
        grid_thw:          &[(u32,u32,u32)],
        deepstack_layers:  &[usize],              // [] 或 [8,16,24]
    ) -> VisionEncodeOutput;

    /// 把含 pad_value 的 input_ids 解析成 hidden_states；下游 GPU kernel 永远看不到 pad_value
    fn resolve_input_embeds(
        &self,
        input_ids: &[u32],
        mm_table:  &MmEmbeddingTable,             // pad_value → vision_embeds slice
    ) -> DeviceTensor;                            // [S, hidden]
}

pub struct VisionEncodeOutput {
    pub vision_embeds:    DeviceVec,
    pub deepstack_embeds: SmallVec<[DeviceVec; 3]>,
}
```

实现矩阵：
- `CudaBackend: VisionBackend` ✓
- `MetalBackend: VisionBackend` ✓
- `CpuBackend` — **不**实现 `VisionBackend`，编译期天然不可作为 VL 后端。
- 测试 mock — 不受影响。

加载时门禁（`model_registry`）：

```
let backend = LoadedInferenceEngine::load(...)?;
if model_arch.is_vision() {
    let _: &dyn VisionBackend = backend.as_vision()
        .ok_or(LoadError::VisionRequiresGpuBackend)?;   // 编译期+加载期双层保险
}
```

`LoadedInferenceEngine::as_vision()` 用 `Option<&dyn VisionBackend>` 暴露，CUDA/Metal 路径返回 `Some`，CPU 路径返回 `None`。

**pad_value 的关键不变量**：
- pad_value 只活在 `IncomingRequest.tokens` 与 `RadixNode.tokens` 两处（用于缓存键与人类调试）。
- 进入任何 GPU kernel 之前，`resolve_input_embeds` 已把它替换成实际 hidden_states；后续 attention/RMSNorm/lm_head/sampler 看到的 token id 永远 `< vocab_size`。
- `lm_head` 输出 logits 的位宽 = `vocab_size`，pad_value 不出现在 logits index 域，sampler 无需 mask。
- 任何对 `input_ids` 做 `as usize` 索引 embed_table 的代码点 → CI 静态扫描禁止（grep `embed_table\[.*input_ids`）。

**Q3-VL DeepStack** 的注入点在 LLM decoder layer 0/1/2 之后；纯文本请求也走带 hook 的 forward（hook 在 `deepstack_embeds.is_empty()` 时跳过）。这条 forward 走 `infer/src/model/qwen3_vl.rs`，不复用 `qwen35.rs`。

### 4.7 ViT 模型实现

CUDA：`crates/cuda-kernels` 新加四个 kernel + 一个模板特化（§5）。Rust 侧在 `infer/src/vision/model/qwen{25,3}vl.rs` 用 backend trait 拼装；不侵入 LLM 现有 forward。

Metal：完全用 `mlx-sys` 既有 + 三处增量 FFI（§6）。**ViT 的 SDPA 调 §7 packed-varlen shader**，传：
- `q,k,v` 为 `[Σ N_patch_in_batch, n_heads, head_dim]`（按 cu_seqlens 段拼接）
- `additive_mask` 形状 `[Σ N_patch, max_seg_len]`，块外位置填 -inf
- `seg_offsets` 给 per-row 起始，复用既有参数；2D-RoPE 在 MSL 端做 axis-lut，不要走 mlx::fast::rope（feedback_mlx_rope_axis 已踩过坑）。

### 4.8 HTTP / OpenAI 协议

支持四种 `image_url.url`：
- `data:image/{png,jpeg,webp};base64,...`
- `http(s)://...`（默认 5s 超时；deny RFC1918+IPv4 loopback+IPv6 ULA/link-local；max_bytes=20MB；content-type 必须 image/*）
- `file://...`（默认拒绝，`--allow-local-image-paths /abs/dir` 白名单后允许；路径必须 canonicalize 在白名单内才可读）
- `blob:` / 其它 scheme（拒绝）

并支持 `image_url.uuid`：客户端 UUID 仅作 **lookup hint**（见 §4.2 `lookup_or_compute`），命中后必须二次校验解码后字节才能复用 mm_hash；不允许跳过哈希。

错误码：
- 400 `invalid_image_format` / `image_aspect_ratio_exceeded` / `too_many_images`（>32）
- 413 `image_payload_too_large`
- 415 `image_mime_unsupported`
- 502 `image_fetch_unreachable`（DNS / connection refused / TLS 失败）
- 504 `image_fetch_timeout`（upstream 超时；不是 408 — 408 语义是客户端慢）
- 503 `vision_preprocessing_overloaded`（preprocessing 内存背压）

返回体头加 `x-arle-mm-hash: <hex>` 让客户端验证缓存命中。

### 4.9 Tokenizer / chat template

Qwen 系列 chat template 已在 `tokenizer_config.json` 内含 `<|vision_start|>{image_pad}<|vision_end|>` 渲染逻辑，但默认要求文本里写 `<|image_pad|>`。我们的 `chat::render_messages_to_prompt()` 检测 `Parts` 中 image-url 时：
1. 在该位置插入 `<|vision_start|>` + `<|image_pad|>` + `<|vision_end|>`（**只一个 image_pad**，保持单图占位）。
2. tokenize 完后用 `expand_image_pad(input_ids, mm_features)` 把每个 `<|image_pad|>` 复制成 `N_llm` 个 `pad_value`（**不是** image_pad 的 vocab id；后者只用于人类可读、tokenizer 阶段；后续走 pad_value）。

**Multi-image 顺序契约**：N 个 `<|image_pad|>` 与 N 个 `mm_features` 按 chat-template **左→右出现顺序** 一一配对；`expand_image_pad` 第 k 个 image_pad 对应 `mm_features[k].mm_hash` / `grid_thw[k]`。配套 unit test：
```
fn pairing_invariant() {
    let prompt = chat::render([msg("look", img_a), msg("compare", img_b, img_c)]);
    let ids    = tokenizer.encode(prompt);
    let pads   = ids.iter().enumerate().filter(|(_,&id)| id == IMAGE_PAD).map(|(i,_)| i).collect::<Vec<_>>();
    assert_eq!(pads.len(), 3);
    let expanded = expand_image_pad(&ids, &[mm_a, mm_b, mm_c]);
    // 第 1 个 pad 区间内全为 mm_a.pad_value，依此类推
}
```

**Startup vocab sanity check**：模型加载后立刻验证 tokenizer 含 `<|vision_start|>=151652`、`<|vision_end|>=151653`、`<|image_pad|>=151655`、`<|video_pad|>=151656`；任一缺失 → 拒绝加载并报 `tokenizer_missing_vision_specials`。旧版 tokenizer 包（pre-2024-08）会缺，必须显式拒绝而不是静默错位。

这样既不破坏 chat template，又避免在 prompt-cache 上对像素内容感知（cache 只看 pad_value 与 mm_hash）。

### 4.10 Weight Loader / Model Registry

`model_registry`：

```
"Qwen2_5_VLForConditionalGeneration" => Vlm(Qwen25Vl)
"Qwen3VLForConditionalGeneration"    => Vlm(Qwen3Vl)
```

`weight_loader.scan_index()` 多返回一个 `vision_shards: Vec<TensorRef>`（`startswith("visual.")`）；语言端 Q3-VL 的 `model.language_model.*` 在 `qwen3-spec` 中加 prefix-strip 别名；GGUF 路径 Phase B（先支持 safetensors，GGUF 用 `mmproj-*.gguf` 兼容 llama.cpp 生态，仅消费方）。

---

## 5. CUDA Kernel Plan

| 项 | 来源 | 工作量 | 验收 |
|---|---|---|---|
| Non-causal varlen attention | FlashInfer 已支持 — 在 `flashinfer_prefill*.cu` 模板中加 `MaskMode::kNone` 实例化；`MaskMode::kCustom` 用于窗口（`additive_bias` 传入） | 新增量 ~200 行；**dispatch 表 + sm-tier 编译 fan-out 影响 5-10 个文件**（FFI / build.rs / sm_*.cu 实例化），列入 §5 末尾的"perf-no-regress 例外清单" | 与 PyTorch `F.scaled_dot_product_attention(..., is_causal=False)` 在 [bf16] 1e-3 内 |
| Conv3d patch_embed | `-lcudnn` + cuDNN cudnnConvolutionForward；spatial=2D, T=2, stride=kernel | ~150 行 wrapper | shape 校验 + 数值对照 HF 1e-3 |
| LayerNorm with bias | 拷贝 `misc/norm.cu`，加 `+ β`；reuse welford | ~80 行 | F.layer_norm 1e-3 |
| GELU exact + GELU tanh-approx | 两个独立 kernel；tanh-approx 严格 `0.5*x*(1+tanh(√(2/π)(x+0.044715x³)))` | ~60 行 | 输出 max-abs vs torch.nn.functional.gelu(x, approximate='tanh') ≤ 1e-3 |
| 2D-RoPE | 复用 1D RoPE，模型代码切 head_dim 两半依次调；**不**改 kernel；Q3-VL 增 `axis_lut: const u8*` 参数（长度 64）选 T/H/W | 1D 端加一个可选指针 ~30 行 | 输出 vs HF apply_multimodal_rotary_pos_emb 1e-3 |
| DeepStack 注入 | `infer/src/model/qwen3_vl.rs` 内 layer 0/1/2 后 hook，调 visual_pos_mask scatter-add；纯文本请求走 hook 但 mask 为空（no-op） | ~150 行（含 hook 接入、mask 构造、跨纯文本/图文路径的统一） | 与 HF Q3-VL 输出 1e-3 |

> Conv3d 备选：im2col + cuBLAS gemm。性能比 cuDNN 略差但避免新链接库 — 留作 fallback feature `--no-cudnn`。

CUDA graph：把 ViT forward 整体加入既有 graph 池。变量维度只有 `Σ N_patch`，按 64 桶量化（实测 ViT FLOPs/N 趋于线性，桶化损失 < 5%）。

**Perf-no-regress 例外**：FlashInfer 模板特化在 P1 阶段允许 ≤5% 文本 prefill 回退（dispatch 表抖动），P2 阶段必须收回到 ≤2%（11.6 上线门槛）。

---

## 6. Metal Kernel Plan

| 项 | 来源 | 工作量 | 验收 |
|---|---|---|---|
| Non-causal SDPA | `mlx_fast_sdpa` 已支持 mask=""；窗口走 §7 packed-varlen shader（既有），传形状不同的 additive mask | ~200 行 Rust 拼装 + 0 行 MSL | 与 PyTorch 1e-3 |
| Conv3d patch_embed | `mlx::conv_general_dilated`：**静态图情形（T=2 同帧）等价于 conv2d，weight 取 `W[0]+W[1]`；视频帧不同情形跑两次 conv2d 后累加**。无须实写 3D 卷积；新增 mlx-sys FFI | ~120 行 FFI + 30 行 Rust 组合 | shape + 数值 1e-3 |
| LayerNorm with bias | `mlx_fast_layer_norm` MLX 已暴露 `weight, bias`；加 mlx-sys FFI 包装 | ~50 行 | F.layer_norm 1e-3 |
| GELU exact / tanh | 用 `mx::erf`、`mx::tanh` 组合；包成 mlx-sys 单调 op，避免在 hot path 多次 launch（fuse via mx::compile） | ~60 行 | 1e-3 |
| 2D-RoPE | **不**复用 mlx::fast::rope（layout 陷阱）；在 §7 shader 内嵌 sin/cos 表 + axis-lut → 一发 kernel 完成 attention+rope | ~120 行 MSL（既有 shader 加分支） | 输出 1e-3 |
| DeepStack | 复用既有 elementwise add；mask scatter 用 mlx::take_along_axis | ~40 行 | 1e-3 |

**Metal 杀手锏**：把 attention + 2D-RoPE + window mask 融成一发 kernel（既有 shader 加 RoPE 分支），跑 Q2.5-VL 同尺寸**目标 ≥ 2× mlx-vlm**（per-window Python loop）；2× 是 §11.3 Metal 验收硬门槛，4× 是预算外彩蛋，不是承诺。

---

## 7. Scheduler 集成 — 状态机

```
state = Admitted
  │  vision tokens budget reservation
  ▼
state = WaitingForEncoder
  │  (every tick)  if encoder_cache hit OR scheduled this tick → goto Prefill
  │                else if rollback budget triggered → keep waiting
  ▼
state = Prefill (chunked)
  │  chunk_range may straddle vision span → vLLM rollback rule applies
  ▼
state = Decode
  │  M-RoPE positions advance per emitted token (text scalar)
  ▼
state = Finished
```

预算反馈环：
- 每个完成的 ViT pass 上报实际 FLOPs；EWMA 更新 `compute_budget_per_step` 的「单位 token 价格」。
- 每个 evict/promote 行为更新 `cache_slots` 软上限（避免 thrash）。

**Q3-VL DeepStack 同步约束**：vision 必须**完全**完成才能开始 LLM layer 0；不能用早期 LLM 层掩盖 ViT 延迟（因为 layer 0 后就要注入）。这一点在 vLLM 现行实现里也是同步的，不是 ARLE 的额外代价。

---

## 8. 多 Agent 协作矩阵

| 阶段 | 任务 | Owner | 并行度 | 交付物 | 交叉 review |
|---|---|---|---|---|---|
| P0 | image preprocessing + hash + mrope（纯 CPU，可单测） | Agent-A (general-purpose) | 单 | `infer/src/vision/{image,hash,mrope,processor}.rs` + 单测 | Codex review (Bash) |
| P0 | weight_loader + model_registry + tensor 名表 | Agent-B (general-purpose) | 与 A 并行 | `weight_loader.rs` 改动 + qwen25vl/qwen3vl 配置 | Codex review |
| P0 | HTTP/OpenAI 解析 + chat template 扩展 | Agent-C (general-purpose) | 与 A、B 并行 | `openai_v1.rs` + `crates/chat` 改动 + http 集成测试 | Codex review |
| P1 | CUDA kernel 增量（LN-bias / GELU / non-causal 模板） | Agent-D (general-purpose, CUDA) | 与 P1-Metal 并行 | `csrc/misc/*.cu` + FFI + golden | Codex review + Plan review (Plan agent) |
| P1 | Metal MLX FFI 增量 + ViT MSL 分支 | Agent-E (general-purpose, Metal) | 与 P1-CUDA 并行 | `mlx-sys` FFI + 既有 shader 改动 | Codex review |
| P2 | Q2.5-VL ViT 装配（CUDA） + 数值对齐 | Agent-D 续 | 串行于 P1-CUDA | `vision/model/qwen25vl.rs`（CUDA path）+ golden 张量 | Codex review |
| P2 | Q2.5-VL ViT 装配（Metal） + 数值对齐 | Agent-E 续 | 串行于 P1-Metal | 同上 Metal | Codex review |
| P3 | 调度器：EncoderCacheManager + RadixCache mm_hash + rollback | Agent-F (general-purpose, scheduler) | 等 P0/P1 收口 | `scheduler/cuda/core.rs`、`prefix_cache.rs`、`backend/metal/scheduler.rs` | Plan review + Codex review |
| P3 | Q3-VL ViT (含 DeepStack + interleaved M-RoPE) 双后端 | Agent-D + Agent-E (并行) | 双后端并行 | qwen3vl 模型 + DeepStack 单测 | Codex review |
| P4 | E2E + bench + 验收 | Agent-G (general-purpose, e2e) | 全部完成后 | 全套 bench + wins 文档 | Plan review (final) |

**Cross-review 规则**：
- 每个 PR 至少经过 **codex review (Bash)** 一次；非 trivial 的（>3 文件 / 跨 backend / 涉数值）再过 **Plan agent** 做 architectural review。
- 数值对齐 PR 必须附 HF transformers 对照脚本 + 输出 max-abs 误差 + 在 wins/ 留 entry。
- 任何 backend isolation 违规（cfg 泄漏、跨 backend type）codex 直接 block。
- **2-strike rule**：一个 subagent 任务 2 次失败 → Claude 亲自接手或换 brief 重派（CLAUDE.md 已锁）。

---

## 9. 风险与回退方案

| 风险 | 概率 | 影响 | 缓解 / 回退 |
|---|---|---|---|
| FlashInfer non-causal 模板特化引入 perf regression | 中 | CUDA 文本 prefill 变慢 | 用独立 instantiation；causal 路径不动；feature `vision-flashinfer` |
| Metal varlen 在大 N_patch 下 occupancy 不足 | 中 | ViT 在 M2 慢 | fallback 到 mlx-vlm 风格的 per-segment SDPA；编译期 feature `vision-metal-loop` |
| cuDNN 链接平台差异（无 cudnn 主机） | 中 | build 失败 | feature `vision-cudnn`（默认 on）+ `vision-cuda-im2col` fallback |
| 大图（max_pixels）引发 OOM | 高 | 单请求挂 | 服务端默认 `max_pixels=1.6M`，超限直接 400 |
| RadixCache 因 mm_hash 节点切分增多导致 lookup 变慢 | 中 | TTFT 抖动 | 引入 mm_run-aware split：节点切割只在 vision span 边界对齐 |
| 多 image 同 prompt 压垮 encoder_cache | 中 | encoder thrash | LRU 改成 SLRU（probationary + protected），且 cache_slots 强约束 |
| Q3-VL DeepStack 注入与 PP/TP 冲突 | 低（当前单卡） | 多卡阶段会复发 | 在 plan：Phase C TP 时把 deepstack scatter 限定在 layer 所在 rank |
| 视频路径需要的 fps/temporal 精度未覆盖 | 必然（Phase B） | 视频不上线 | 显式 Phase B；image 上线先 |

---

## 10. Phase 时间表（按现有团队节奏 = ckl + claude + 平行 subagent）

| Phase | 范围 | 退出条件 |
|---|---|---|
| **P0**（1 周） | image/hash/mrope + HTTP 协议 + weight loader + tokenizer 扩展 + Backend trait 接口（mock 实现） | http 测试：上传一张图，返回 image_pad 已展开的 input_ids 与 mm_hash；HF processor 数值 1e-6 |
| **P1**（1 周） | CUDA + Metal kernel 增量；ViT 单层 forward 各自跑通 | 单层 vit 输出 vs HF 1e-3，bench 单层延迟落 wins/ |
| **P2**（1.5 周） | Q2.5-VL 完整 ViT + merger + LLM 替换占位；CUDA & Metal 端到端文本输出 | 跑 Qwen2.5-VL-7B "what's in this image" 输出与 HF 同 prompt 在 top-1 token 一致；guidellm 跑通 |
| **P3**（1.5 周） | Q3-VL（含 DeepStack + interleaved M-RoPE）+ EncoderCacheManager + RadixCache 注入 | 多图多请求并发 1k req 不 OOM；同图重发 TTFT 降至 ≤ 0.3× 首发 |
| **P4**（1 周） | bench / 验收 / 文档 / wins 全套；远端 SM-tier 矩阵 | §11 验收全过 |

**Phase B**（不在本计划首期）：视频、Qwen3-Omni 音频、GGUF mmproj、TP/PP 多卡、跨多轮对话的 mrope `current_pos` 持久化（agent CLI 同图换 prompt 已被 §1.3 (0,1) 命中场景吃掉）。

---

## 11. 验收方案（Acceptance）

> 所有验收必须落 `docs/experience/wins/` 的对应 entry；每条都标注 owner 与可复现命令。

### 11.1 单元数值（必过；门：max-abs 误差）

| 项 | 输入 | 参考 | 阈值（bf16） |
|---|---|---|---|
| smart_resize / pack_patches | 9 张 fixture（极小、非整除、超 max_pixels、宽高比 199、EXIF rot 8 等） | HF Qwen2VLImageProcessor | 像素 1e-6（确定性） |
| Conv3d patch_embed | 上面输出 | HF `visual.patch_embed` | 1e-3 |
| 单层 ViT block (full attn) | 随机 [N=1024, hidden] | HF block forward | 1e-3 |
| 单层 ViT block (window attn, Q2.5-VL) | 同上 + cu_window_seqlens | HF block forward | 1e-3 |
| 2D RoPE (Q2.5-VL & Q3-VL interleaved) | 随机 q,k + grid_thw | HF apply_multimodal_rotary_pos_emb | 1e-3 |
| Patch merger (RMSNorm + GELU 严格) | 随机 [N, 5120] | HF merger | 1e-3 |
| Patch merger (LayerNorm + GELU tanh) | 随机 [N, 4608] | HF merger | 1e-3 |
| DeepStack 注入 | 随机 hidden_states + 3 个 deepstack | HF `_deepstack_process` | 1e-3 |
| M-RoPE position_ids | 输入 token + grid_thw 列表（含混合图文） | HF `get_rope_index` | 完全相等（int） |

### 11.2 端到端正确性（必过）

bf16 跨实现 1ulp 漂移在 30+ token 后偶发分叉，硬等 64 token 不现实。指标改成两段式：

- **Image-only QA 11 道**（greedy，CUDA & Metal 各跑）：
  - 前 16 token 与 HF transformers 完全一致。
  - 前 64 token 中 logit-rank-1 重合率 ≥ 98%（即每步 argmax 与 HF 不同的步数 ≤ 1/50）。
  - 出现 tie 的 step（top-1 与 top-2 logit 差 < 1e-3）必须用同一 tie-break 规则（按 token id 升序）。
- **Multi-image 5 道**：1 prompt 含 ≥2 张图，前 16 token 一致 + 重合率 ≥ 95%。
- **超长图文 3 道**：max_pixels 边界 + 8k 文本，前 8 token 一致 + 不 OOM。
- **MMMU 子集 200 题**：Qwen2.5-VL-7B vs HF reference accuracy 偏差 ≤ 1%（CUDA）/ 2%（Metal）。

### 11.3 性能基线（强制 wins entry；guidellm）

`scripts/bench_guidellm.sh vision-q25vl-cuda-h100`：
- 单图 chat (1 image @ ~1.5k vision tokens + 200 text in / 200 out)：
  - CUDA H100 `Qwen2.5-VL-7B`：**TTFT ≤ 200ms** @ c=1；**throughput ≥ 1.4× 现行 Qwen3 同尺寸文本** @ c=16（因为 ViT 平摊）；**hit ratio ≥ 95%** 当 100 req 复用同图。
  - CUDA RTX 4090 同上比例容忍 0.7×。
- Metal M2 Pro：TTFT ≤ 1.5s 单图（demo grade）；throughput 不做硬指标，但比 mlx-vlm 同输入 ≥ 2×（D5 兑现门槛）。
- 远端机器跑不到的：本机 commit 留 `pending-remote` stub，远端跑后回填 wins entry（CLAUDE.md 强制）。

### 11.4 跨后端等价（必过）

两后端均 bf16；CUDA 与 Metal 同 prompt 同图 greedy 前 16 token 完全一致，第 17-64 步 argmax 不同的 step ≤ 2。提供一键脚本 `scripts/parity_check_vision.sh`，跑 20 case，每条结果落 wins/。

### 11.5 鲁棒性 / 异常（必过）

| 输入 | 期望 |
|---|---|
| 32×32 PNG | 接受（≥ min_pixels 后展开） |
| 8K JPG（> max_pixels） | 服务端 clamp 后 200 OK，response header 标注下采样比 |
| EXIF orientation 8 | 正确旋转 |
| EXIF `0xA420` ImageUniqueID 重复但像素不同 | **不**别名；二次像素校验生效，两条不同 mm_hash |
| GIF / SVG / TIFF / BMP | 415 |
| http upstream 超时 | 504（不是 408） |
| http upstream 拒连/DNS 失败 | 502 |
| RFC1918 / IPv6 ULA / link-local URL | 400 SSRF refused（除非 `--allow-private-image-urls`） |
| `file://` 跳出白名单 | 400 |
| 100MB 图 | 413 |
| 1000 张图同 prompt | 400 too_many_images（默认上限 32） |
| 单请求累计 decoded 像素 > 256 MiB | 413 |
| 进程总 preprocessing 内存压满 | 503 + Retry-After |
| `image_url.uuid` 提供但字节不同 | 重新计算 mm_hash，**绝不**复用 hint 缓存项 |
| tokenizer 缺 `<|image_pad|>` | 加载阶段拒绝 + `tokenizer_missing_vision_specials` |

### 11.6 上线门槛（任一不达 → 不发版）

- 11.1 全过；11.2 image-only + multi-image 全过；11.4 通过；11.5 全过。
- guidellm bench 入 wins/ 至少 1 个 CUDA + 1 个 Metal entry；CUDA throughput 在既有 Qwen3 纯文本场景回退 ≤ 2%（FlashInfer 模板特化的 perf-no-regress 例外清单已在 §5 末尾收口）。
- Metal：跑 mlx-vlm 同输入对照，Q2.5-VL-7B M2 Pro **≥ 2× mlx-vlm**（D5 兑现）。
- `cargo clippy --all-features -- -D warnings` clean；`cargo test --workspace --release` green；CI 的 `grep -nE 'embed_table\[.*input_ids'` 无命中（pad_value 漏检静态门）。
- HTTP fuzz：1000 个随机 image_url payload 不出现 panic（含 SSRF / oversize / malformed EXIF / truncated JPEG）。
- 安全等价：同 image bytes 不同 EXIF UUID hint → 同 mm_hash；不同 image bytes 同 hint → 不同 mm_hash（多租户隔离回归测试）。
- 内存：连续 1h 1k 不同图像 chat，RSS 增长 ≤ 200MB（encoder cache 上限生效）；preprocessing 工作集峰值 ≤ 8 GiB。

---

## 12. 一句话回顾

**ARLE 视觉 = 单进程 Rust + vLLM v1 双预算调度 + 两层正交缓存（RadixCache 吃前缀+图像，EncoderCache 吃 ViT）+ pad_value 解析层守住 token-id 边界 + 独立 `VisionBackend` 子 trait 让 CPU 后端编译期天然不实现 + Metal-first ViT shader 复用 §7 packed-varlen。零新 crate；客户端 hint 永不替代规范哈希；DeepStack 在 (1,0) 缓存场景下强制 ViT 重跑。**
