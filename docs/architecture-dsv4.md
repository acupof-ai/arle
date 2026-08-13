# DSv4 Architecture — Prefill / Decode Paths & Kernels

Canonical mechanism-level map of the **DeepSeek-V4-Flash / GLM-5.2** CUDA
inference path (`crates/infer-cuda/`). Covers every prefill and decode path, the
kernels each dispatches (vendored vs hand-rolled), and the four hardest
subsystems (FlashMLA decode, DeepGEMM grouped-MoE, DSA sparse indexer, MTP
spec-decode). Ends with a survey of the latest spec-decode practice
(**DeepSeek DSpark**, 2026-06-27) and where it maps onto our runtime.

> **Line numbers are point-in-time snapshots (2026-06-29 `main`).** They drift
> across checkouts. Re-grep the **symbol names** (stable anchors) before
> trusting an exact line: e.g. `forward_tokens_stream_impl`,
> `try_flashmla_decode_attention`, `deepgemm_grouped_experts`, `csa_select`,
> `commit_accepted_fold`, `kv_budget_num_slots`, `flashmla_device_page_table`.

Companion docs: [architecture.md](architecture.md) (crate boundaries),
[codebase-map.md](codebase-map.md) (where to start reading),
[support-matrix.md](support-matrix.md) (model/quant tiers).

Two CUDA model families ride this path, selected by `model_type` /
`hc_mult`:

| Tag | Model | `model_type_int` | `d_qk` | KV pool B/tok | `hc_mult` |
|-----|-------|------------------|--------|---------------|-----------|
| **MODEL1** | DeepSeek-V4-Flash | 1 | 512 | **584** | >1 (hyper-connection) |
| **V32** | GLM-5.2 (`glm_moe_dsa`) | 0 | 576 | **656** | 1 (plain residual) |

---

## 0. Top-level dispatch — who routes prefill vs decode

`Dsv4Model::forward_tokens` → `forward_tokens_impl` (`dsv4.rs:1882 / 1917`) is the
single-sequence entry. **Prefill and eager-decode share one function**
(`forward_tokens_stream_impl`, `dsv4.rs:4661`), forking on `seq_len` (token count)
and `start_pos_device` (filled only when `seq_len==1`).

```
forward_tokens_impl (dsv4.rs:1917)
└─ forward_tokens_stream_impl ← prefill (seq_len>1) AND decode (seq_len==1)

forward_decode_batch (dsv4.rs:2125) → forward_decode_batch_stream_impl (dsv4.rs:2175)
 → batched decode lane, MODEL1-only, concurrency lever #60 → decode_lane_fwd (dsv4.rs:3275)

forward_tokens_verify_scheduled (dsv4.rs:2054) ← MTP spec-decode chain verify (frozen)
forward_tokens_verify_stream_persistent (dsv4.rs:4351)
forward_decode_batch_verify (dsv4.rs:3858)
```

Per layer, `forward_tokens_stream_impl` runs two HC-wrapped halves:
**Attention half** (HC-pre-norm → MLA → O-LoRA all-reduce → HC-post) and
**MoE half** (HC-pre-norm → dense MLP *or* routed MoE + shared expert → HC-post).
For MODEL1 `hc_mult>1` the HC mixers run (`hc::gen_mhc_params`, `mhc_pre_rms_norm`,
`hc_post`); for V32/GLM `hc_mult==1` they collapse to plain `rms_norm_batch` +
`add_batch`. LM-head tail is shared: `forward_stream_last_token` (`dsv4.rs:4285`).

---

## 1. Prefill path (`seq_len > 1`, `start_pos_device = None`)

### 1.1 Pre-layer (once)
`embedding_batch` (`dsv4.rs:4729`) → HC residual-stream expand
`hc::initial_stream_from_embeddings` (`dsv4.rs:4738`, `stream_dim = hidden*hc_mult`;
identity for GLM).

### 1.2 Attention half — MLA
`mla_attention` (`attention.rs:8005`) = `mla_attention_prepare` (projections /
RoPE / indexer) + `mla_attention_fwd` (the attention kernel).

- **Projections (GEMM)** in `mla_attention_prepare` (`attention.rs:8935`): `wq_a`
 → `q_norm` (RMSNorm) → `wq_b`; `wkv` → `kv_norm`. All via `dsv4_linear`
 (`attention.rs:5740`: FP8 block-scaled GEMV / DeepGEMM / bf16 cuBLAS). Prefill
 fusion: `run_fused_wqkv_prefill` (`attention.rs:5531`), `prefill_proj_deepgemm`
 (`attention.rs:5376`).
- **RoPE (partial dims + Hadamard)**: `dsv4_prepare_qk_cuda` (`attention.rs:9220`;
 kernel `dsv4_prepare_qk_fused_kernel`, `csrc/misc/dsv4_attention.cu:274`,
 hand-rolled).
- **DSA indexer** (sparse key selection, mode-gated): see §4.
- **Write KV (bf16 pack)**: `arle_flashmla_csa_pack_kv` (`attention.rs:6358`;
 hand-rolled `csrc/misc/arle_flashmla_csa_prep.cu`) packs one contiguous bf16
 pool `[SW ring | current-chunk K | compressed pool]`.
- **Build sparse indices** (per mode): `arle_flashmla_{chain_verify,csa,hca}_build_indices`
 (`attention.rs:6406/6438/6478`, hand-rolled same file).
- **Prefill attention kernel** = `arle_flashmla_sm90_sparse_prefill_fwd`
 (`attention.rs:6620`; shim `csrc/misc/arle_flashmla_shim.cu:38`) →
 **vendored FlashMLA `sm90::run_fwd_kernel`** (`vendor/flashmla/csrc/sm90/prefill/sparse/fwd.cu`,
 sparse varlen prefill, B_H=64 / B_TOPK=64).
- **Tail**: TP repack/out-slice (`dsv4_tp_q_repack_cuda` / `dsv4_tp_out_slice_cuda`),
 output inverse-RoPE (`arle_dsv4_output_inverse_rope_batch_*`), SW ring update
 (`dsv4_update_window_cache_cuda`, `csrc/misc/dsv4_attention.cu:863`).
- **O-LoRA all-reduce**: `tp.all_reduce_sum` (`dsv4.rs:4909`, NCCL, single-GPU no-op).

### 1.3 MoE half
- **GLM dense layer** (only `per_layer_dense_mlp[i]`): `dsv4_dense_mlp_forward`
 (`dsv4.rs:6228`) — bf16 SwiGLU FFN, `dsv4_linear` gate/up → `ops::silu_mul`
 (`csrc/misc/elementwise_basic.cu:86`) → `dsv4_linear` down. (DSv4 is MoE on
 every layer; this is GLM-only.)
- **Routed MoE**: router GEMM (`gemm_batch(&layer.gate)`) → `dsv4_route`
 (`moe.rs:2602`; kernel `dsv4_route_kernel`, `csrc/moe/dsv4_route.cu:329`,
 hand-rolled). DSv4 uses **sqrtsoftplus scoring (kind=2) + NoAuxTc / LearnedBias**
 (`e_score_correction_bias` steers selection; emitted weight uses the unbiased
 score). `n_group/topk_group` group-limited routing exists only on the host
 fallback (`infer-moe/src/route.rs:209`); DSv4-Flash sets neither, so the device
 kernel always runs. Transport selection + grouped GEMM detail in §3.
- **Shared experts (always-on)**: `dsv4_shared_expert_forward` (`moe.rs:3696`),
 added at `dsv4.rs:3756`.

---

## 2. Decode path (`seq_len == 1`)

Three physical lanes share the same kernels. **B=1 always takes the eager
single-row lane**: `forward_decode_batch` early-returns to the single-row path
when `rows.len()==1` (`executor.rs:2798`) — the batched lane never executes at
B=1.

### 2.1 Eager decode (`forward_tokens_stream_impl`, `seq_len==1` branch)

**Per-token collectives at TP>1 = 3 per layer** (nothing per-step): the Q
head-slab all-gather (`attention.rs:2750`), the attn O-LoRA all-reduce
(`dsv4.rs:5055`), and the MoE all-reduce (`dsv4.rs:5308`). lm_head is
replicated (full-vocab GEMV per rank, `dsv4.rs:6286`) and sampling is
rank-local, so the step total is exactly `3 × num_hidden_layers`.
MODEL1 (`w_kc/w_vc/o_proj` all `None`) → `mla_attention_decode_graph`
(`dsv4.rs:4859`, eager call does not capture); V32/GLM → `mla_attention`
(`dsv4.rs:4881`). Attention core = `try_flashmla_decode_attention`
(`attention.rs:6746`):

1. **Write KV (FP8 pack)**: `flashmla_pack_sw_ring` / `flashmla_pack_one_sw_token`
 / `flashmla_pack_compressed_delta` (`attention.rs:6819/6824/6836`, hand-rolled
 `csrc/attention/dsv4_fp8_kv_pack.cu`).
2. **Read-side page table**: `pool.flashmla_device_page_table(slot)`
 (`attention.rs:6870`). **Fixed**: eager decode now routes the device page
 table (the historic `None` at ~`attention.rs:6496` is stale — 6496 is now in
 the prefill function body).
3. **Build decode indices**: `dsv4_flashmla_decode_build_indices_start_pos_ptr`
 (`attention.rs:6875`; kernel `csrc/attention/dsv4_flashmla_decode_build_indices.cu:186`).
4. **Decode attention kernel** = `arle_flashmla_sm90_sparse_decode_fwd`
 (`attention.rs:7016`; shim `csrc/misc/arle_flashmla_decode_shim.cu:209`) →
 **vendored FlashMLA `sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel`
 + `run_flash_mla_combine_kernel`** (SM90 sparse-FP8 split-KV decode).

Eager fallback (FlashMLA decode off): hand-rolled fused MLA cores
`dsv4_swa_attention_start_pos_ptr_cuda` (SW, `csrc/misc/dsv4_attention.cu:751`) /
`dsv4_hybrid_attention_start_pos_ptr_cuda` (CSA/HCA, `.cu:1765`).

MoE half at decode — **default transport is `allreduce`** (`ARLE_DSV4_MOE_TRANSPORT`
unset ⇒ local routed experts + per-layer TP all-reduce, `dsv4.rs:5308`); DeepEP-LL
is opt-in:
- LL transport (opt-in) = DeepEP `internode_ll` dispatch/combine (NVSHMEM
 IBGDA, **FP8 e4m3 packed in-flight**).
- Small-batch bypass: `total_routes ≤ 8` → `dsv4_moe_forward_decode_fp8`
 (`moe.rs:2918`), a hand-rolled warp-per-row w8a16 grouped GEMV (not DeepGEMM).
- **Comm-overlap**: shared expert runs on `comm_stream` concurrent with the
 routed all-reduce (`dsv4.rs:4989/5140`, pipeline fence).

### 2.2 Batched decode lane (`forward_decode_batch_stream_impl`, MODEL1-only, lever #60)
`decode_lane_fwd` (`attention.rs:2510`): `build_indices_batched` (per-row page
table) → `sched_meta_for_batch` → **one** batched
`arle_flashmla_sm90_sparse_decode_fwd` over n rows (`attention.rs:2218`). This is
the **21→76 slot concurrency payoff** executor (commit `5352e247`, TP=4/EP=4,
max_seq=16384, ~3.62×).

### 2.3 LM-head tail (all lanes)
`forward_stream_last_token` (`dsv4.rs:4285`): last-token wide-stream row → head HC
fold (MODEL1) / `copy_row_to_vec` (GLM) → final `rms_norm_vec` →
`lm_head_project` (GEMV, replicated full-vocab per rank) → `sample_cuda_token`
(the NON-scratched sampler: greedy argmax allocs a 1-int scratch every token,
`ops.rs:432` — unlike Qwen3.5's zero-alloc `sample_cuda_token_scratched`).

### 2.5 Prefill vs decode quick-diff

| Dim | Prefill (`seq_len>1`) | Decode (`seq_len==1`) |
|-----|-----------------------|------------------------|
| Position metadata | host `start_pos` | device `start_pos_device` (capture-safe) |
| KV write | bf16 whole-pool pack `arle_flashmla_csa_pack_kv` | FP8 incremental pack `dsv4_fp8_kv_pack.cu` (page-routed) |
| Attention kernel | FlashMLA **sparse prefill** `sm90::run_fwd_kernel` | FlashMLA **sparse-FP8 split-KV decode** `run_flash_splitkv_mla_fp8_sparse_kernel` |
| MoE transport | intranode all-to-all / LL owned-slice batched | **LL internode_ll** (FP8 in-flight) / small-batch GEMV |
| Grouped GEMM layout | allreduce=contiguous, deepep=masked (both vendored DeepGEMM) | pooled masked / small-R hand GEMV |
| Comm overlap | — | shared-expert ∥ routed all-reduce |

### 2.6 Page-attn / page-tier identity (2026-06-30)

DSv4 now connects to the same host page identity flow as other CUDA models, but
with DSv4's fixed-band semantics:

- `infer-seam::HostPagedKvPool::set_fixed_pages_per_slot(pages)` makes host
 allocation draw the whole logical FlashMLA band once per slot. `truncate_slot`
 only moves the logical cursor in this mode; it never frees tail band pages.
- `KvBatchDescriptor` carries both `flat_page_ids` (live token prefix) and
 `flat_slot_page_ids` (complete slot page table). Sequential models keep using
 `page_range`; DSv4 lowers `slot_page_range`.
- `Dsv4KvAdapter::prepare_kv_batch` mirrors `flat_slot_page_ids` into every
 layer's `TokenKVPool` via `mirror_band`, then advances the FlashMLA cursor.
 FlashMLA prefill/decode pack and read paths therefore resolve through the
 engine/radix/tier page identity rather than `slot * fixed_band` arithmetic.
- Whole-slot restore and position-0 prefix restore receive the host slot page
 table from `infer-core` and mirror it before copying `Dsv4SlotSnapshot` payloads
 back to device memory.
- TP support is rank-local bytes + TP scalar consensus: each rank stores its own
 shard image under the same engine key; hit length, demote room, image-fit, insert,
 read/parse/restore success are all reduced with `TpRuntime::all_reduce_min_scalar_i32`.
 Any rank miss/failure makes every rank take the same recompute/error branch.

The page-granular radix tier remains dense-Qwen-only until the DSA sidecar is
itself page-addressable at arbitrary radix boundaries. DSv4's safe reuse route is
position-0 snapshots plus fixed-band page-table restore; whole-slot
capacity spill uses the same rank-local image protocol.

---

## 3. FlashMLA decode core — split-KV + combine (vendored FlashMLA)

### 3.1 Why split-KV
Decode is `s_q=1` one query token but `h_q=64/128` query heads against a long
sparse KV list (`topk_unified` selected tokens). One CTA per request leaves SMs
idle → cut each request's KV-block range into contiguous **splits**, one per
SM-partition (z-grid CTA), then combine partial softmaxes (flash-decoding).

### 3.2 How many splits (data-dependent)
- `num_sm_parts` upper bound = `max(num_sms / s_q / (h_q/64), 1)` (vendored
 `api/sparse_decode.h:59-66`; ATen-free shim replica `arle_flashmla_decode_shim.cu:120-132`).
 H20 h_q=128 ≈ `num_sms/2`, h_q=64 ≈ `num_sms`.
- Actual split by `get_mla_metadata_kernel` (`get_decoding_sched_meta.cu:30-126`,
 single warp `<<<1,32>>>`): per-request blocks = `ceil(topk/64) + 5`
 (`+5` = `fixed_overhead_num_blocks`, prevents over-splitting tiny requests),
 greedily partitioned across `num_sm_parts`, `payload = ceil(total/parts)+5`.
 **Long context (large topk) → more blocks → more splits → SMs saturated.**
 `num_splits_ptr` is the per-request split prefix-sum; `num_splits[b]==1` → combine
 early-exits.

### 3.3 The kernel (`splitkv_mla.cuh:86-678`, vendored)
- **Grid** = `(NUM_M_BLOCKS, s_q, num_sm_parts)`, **block=384 = 3 warpgroups**.
 h_q=128 → `NUM_M_BLOCKS=2` form a Hopper cluster sharing dequantized K via DSMEM.
- **Warp specialization**: WG2 = producer (walks topk blocks, gathers selected
 FP8 tokens, dequantizes bf16 into double-buffered smem); WG0/WG1 = consumers
 (QK WGMMA → online softmax → P@V, V's 512-wide latent split across the two WGs).
- **Math**: QK `MMA_64x64x16_F32BF16BF16`; softmax `exp2f` (scale folded into
 `sm_scale_div_log2 = sm_scale*1.4427`); PV `MMA_64x256x16`; `sV` aliases `sK`
 smem (MLA K==V latent).
- **Sparse**: producer reads `gIndices` (DSA-selected pool-absolute slot ids);
 `token_index==-1` = mask sentinel; MODEL1 also enforces per-request
 `topk_length`.

### 3.4 FP8 pool layout (dequant in-kernel, no separate scale arg)
- **MODEL1 (584 B/tok)**: 64-token AoS body `[448 fp8 NoPE + 128B bf16 RoPE]=576B`
 + trailing 8B e8m0 scale region/token; in-kernel `__nv_cvt_e8m0x2_to_bf162raw`
 decodes 8 power-of-two scales (one per 64 dims).
- **V32/GLM (656 B/tok)**: inline `[512 fp8 NoPE][4×f32 scale][128B bf16 RoPE]`;
 4 f32 scales (one per 128 dims). **V32 forbids dynamic `topk_length`/two-tier**
 (in-kernel assert).
- Write-side pack (`dsv4_fp8_kv_pack.cu`): MODEL1 uses e8m0 `ceil(log2(amax/448))`;
 V32 switched to inline f32 `amax/448` (e8m0 pow2 rounding costs up to ±40%/block).

### 3.5 Combine (`combine.cu:18-162`, vendored)
Grid `(b*s_q, 1, ceil(h_q/8))`, one warp per head. `my_num_splits==1` returns
immediately (the "no split → skip combine"). LSE merge:
`max_lse → sum=Σexp2(lse-max) → global=log2(sum)+max`, then `scale=exp2(local-global)`
weighted-sums the O partials → bf16 `out`. Uses PDL
(`programmaticStreamSerializationAllowed`) to chain after the MLA kernel without
explicit sync.

### 3.6 Single-row vs batched
Same shim entry, same kernel; only `b` differs (b folds into the **split index**,
not a stride). Single-row computes sched-meta once at slot init
(`init_constant_sched_meta`, capture-safe); batched recomputes per forward with
`b=n` (`sched_meta_for_batch` — the b=1 cached meta is wrong for n>1, a #60
pitfall). Batched is **MODEL1-only** (asserts `head_dim==512`, bakes 584B
strides) — V32/GLM must use single-row. The lever replaces a per-row launch loop
with one kernel launch over N rows.

---

## 4. DSA sparse indexer — paged MQA logits scoring

### 4.1 Concept + config
The lightning-indexer scores all causally-visible past tokens cheaply with a
small **MQA** head in FP8, then top-k selects `index_topk` keys for the main MLA
to attend over. `score_scale = index_head_dim^-0.5 * index_n_heads^-0.5`.
Config (`deepseek-spec/src/{v4,glm}.rs`): `index_n_heads / index_head_dim /
index_topk`. Runtime **asserts `head_dim==128`, `n_heads∈{32,64}`**; topk is
config-driven (GLM-DSA 32/128/2048, a DSv4 fixture 64/128/512).

### 4.2 Index-key cache WRITE (per new token)
- **Hadamard rotation** (`hadamard128_bf16_kernel`, `dsv4_dsa_official.cu:196`,
 vendored port): orthonormal transform spreads energy across 128 dims, preserving
 `q·k`, so FP8 quant doesn't get crushed by one large coordinate. Impl: 2 intra-
 thread radix-2 butterflies + 5-step `shfl_xor` warp Walsh–Hadamard + `rsqrt(128)`.
- **FP8 store** (`fused_store_indexer_cache_kernel`, `.cu:334`): page=8448B/64slot,
 per-slot `[128B fp8 key][4B f32 scale]=132B`; `page=index>>6, offset=index&63`.
- **Fixed-band sidecar — NOT in the FlashMLA page pool**:
 `Dsv4LayerKvLayout.dsa_key_cache` (`attention.rs:262`, FP8, full history, what
 the scoring kernel reads), summed as `state_caches_per_slot` in
 `kv_budget_num_slots` (`dsv4.rs:1645`). The FP8 cache still grows linearly with
 `max_seq` and is restored as part of `Dsv4SlotSnapshot`; it is not yet a
 page-granular radix-tier object. The bf16 `rotated_keys` is **no longer** a
 full-history mirror: as
 of 2026-06-29 it is a transient drain-immediate staging ring
 (`dsv4_dsa_rotated_ring_rows`, capped at `DSV4_INDEXER_STAGING_RING_ROWS`),
 removing its O(max_seq) per-slot term (−254 MiB/slot/layer at 1M) — see
 [`wins/2026-06-29-dsv4-dsa-rotated-key-transient-ring.md`](experience/wins/2026-06-29-dsv4-dsa-rotated-key-transient-ring.md)
 (pending-remote needle-gate).

### 4.3 Index query + score + top-k READ
- **Build indexer Q** (`fused_q_indexer_rope_hadamard_quant`, `.cu:101`, vendored
 port): one kernel fuses RoPE (rope lanes only) + Hadamard + FP8 quant; per-row
 scale folded into `weights_out`.
- **Paged FP8 MQA logits** (`dsv4_deepgemm_fp8_paged_mqa_logits_fused_cache_cuda`,
 `deepgemm_native.cu:1990`, **vendored DeepGEMM template, JIT**): computes
 `Σ_h weights[r,h]·(q_fp8[r,h]·k_fp8[j])` per (query, past-token) — **paged** via
 `block_table` (block_kv=64). Metadata kernel (`smxx_paged_mqa_logits_metadata`,
 split_kv=256) balances variable-length KV work across SMs.
- **Top-k** (`deepseek_v4_topk_transform_kernel`, `.cu:634`, vendored port):
 `seq_len≤topk` → naive all-select; else `radix_topk` (MSB-first radix selection,
 histogram+cumsum, 4 refine rounds, no full sort). Selected positions →
 `page_to_slot` → `selected` (slot-relative page indices).

### 4.4 How `selected` connects to FlashMLA
`selected` does **not** go straight into the kernel; it passes through
build-indices, which merges sparse selection with the sliding-window blocks and
translates to pool-absolute:
- Decode: `dsv4_flashmla_decode_build_indices_start_pos_ptr` (`attention.rs:6875`)
 consumes `selected_ptr_u64` + SW + page table → `scratch.indices`.
- Prefill: `arle_flashmla_csa_build_indices` (CSA) / `hca` / `chain_verify`.
- Batched: `selected_batched` → `build_indices_batched` → `self.indices[n, topk_unified]`.
- `topk_unified = sliding_window + chain_pad + max_compressed_keys`,
 `max_compressed_keys = index_topk` → FlashMLA attends `[SW blocks ∪ index_topk
 selected keys]`.

### 4.5 Compressor vs indexer (mode-dependent)
Two separate mechanisms, both per-token:
- **Compressor** (`compressor_forward`, `attention.rs:11640`; `dsv4_compressor_*`
 kernels in `dsv4_attention.cu`) **builds** the compressed latent KV (folds every
 `compress_ratio` tokens into one row) that FlashMLA attends over.
- **Indexer (DSA)** **selects** which keys to attend.

| Mode | Compressor | Indexer | Notes |
|------|-----------|---------|-------|
| **CompressedSparse (DSv4-Flash CSA)** | ✅ | ✅ | both on; CSA even compresses its index keys |
| **SparseIndexed (GLM-DSA)** | ❌ | ✅ | indexer-only over full-res latent (ratio=1) |
| HybridCompressed | ✅ | ❌ | compressor only |
| SlidingWindow | ❌ | ❌ | neither |

---

## 5. DeepGEMM FP8 grouped MoE — masked vs contiguous + SM90 JIT

### 5.1 The two layouts
- **Contiguous**: flat `[m,K]` FP8 activations, `m_indices[row]` names each row's
 expert (`-1`=pad). Kernel resolves the B group **once per `BLOCK_M` tile** from
 the tile-start row → each expert segment must be `BLOCK_M`-aligned
 (`DEEPGEMM_CONTIG_ALIGN=128`, decode band 64); pad rows skipped via
 `is_computation_valid >= 0` (`scheduler/gemm.cuh:285`).
- **Masked**: `[E, m_padded, K]` band, `masked_m[e]` = real rows/expert. Two skips:
 scheduler enumerates only `ceil(masked_m[e]/BLOCK_M)` M-blocks (empty tail blocks
 never scheduled), and the partial-block row skip `row < masked_m[e]`
 (`gemm.cuh:287`).

**Why two**: contiguous → prefill / large route counts (masked's unpad work
`32*T*topk*H` overflows i32 at ~1.5K prompt tokens; contiguous packs only
`~total_routes` rows). Masked → decode / DeepEP-LL — the win is **grid-trimming
the quant kernels** (`silu_mul_masked_quant` covers only `expected_m` rows; a
631µs/layer empty-block drain = 52.9% of deepep_ll GPU time at B=1, collapsed
~2048×). Selection: `use_masked = total_routes ≤ 128` (`moe.rs:1210`).

### 5.2 JIT pipeline (`deepgemm_native.cu`)
`generate_kernel_code` (`:1177`) emits an instantiation of `sm90_fp8_gemm_1d2d_impl`
(template params: `BLOCK_M/N/K`, `num_stages`, `num_groups`,
`GemmType::{Normal,MGroupedContiguous,MGroupedMasked}`). `get_best_config` (`:671`)
picks a layout via an L1/L2 cycle model. **`nvcc -cubin -O3 -std=c++20
--gpu-architecture=sm_90a`** (not nvrtc; c++20 fixes a gcc-13 libstdc++ `requires`
→ `CUDA_ERROR_UNKNOWN`). Cached at `~/.deep_gemm` (FNV key, cross-process `flock`,
atomic rename); `cuobjdump -symbols` → `cuModuleLoad` → `cuLaunchKernelEx`.
**SM90-only** (`prop.major != 9 → NOT_SUPPORTED`).

### 5.3 The vendored 1d2d kernel (`sm90_fp8_gemm_1d2d.cuh:48`)
"1d2d" = **1-D block scales on LHS** (A/activations, per-128-channel-along-K, one
per row), **2-D block scales on RHS** (B/weights, 128×128 tile grid). TMA
warpgroup drives the `num_stages` pipeline (A fp8 / SFA f32 / B fp8 copies +
`arrive_and_expect_tx`); math WGs run WGMMA (`FP8MMASelector<BLOCK_N>`).
**Dequant fold**: `final_accum += (scale_a * scale_b) * accum` across K-blocks.
Epilogue STSM+TMA store bf16. Persistent scheduler, one CTA/SM, L2-reuse block
reorder.

### 5.4 The FP8 quant dance (`dsv4_deepgemm_ops.cu`, hand-rolled)
- `pack_quantize_bf16_to_fp8` (`:63`): per-128-block `scale=block_max/448`, f32
 column-major (matches SFA TMA), e4m3 cast.
- `swiglu_quantize_w13` (`:120`): fused **clamped SwiGLU** (`dg_swiglu`:55 —
 `gate=min(gate,limit); up=clamp(up,±limit); silu(gate)*up`) on the w13 output +
 per-128 requant for the w2 GEMM.
- `silu_mul_masked_quant` (`:191`, LL 3-D path): touches only `expected_m` rows;
 out-of-bound `__trap()`s loudly rather than silently dropping rows.

### 5.5 Scale format
**DSv4 MoE uses plain f32 block scales on SM90, NOT UE8M0** (UE8M0 is the SM100
packed format; only referenced in a k-grouped scheduler comment). Evidence:
`float* scales`, SFA TMA `FLOAT32`, `GroupedCache.scales: f32`. (Distinct from the
attention KV pool, where MODEL1 uses e8m0.)

---

## 6. MTP speculative decode — chain verify + rollback snapshot

### 6.1 Draft head
`Dsv4MtpLayer` (`dsv4.rs:331`) = a full `Dsv4Layer` (attn + MoE + two HC mixers) +
MTP input-combine/output-head tensors (`enorm/hnorm/e_proj/h_proj/head_hc/norm`).
Loaded only when `spec_decode_on && num_nextn_predict_layers>0`; asserts exactly
one nextn layer; forced `compress_ratio=0` (SlidingWindow). Proposal
`mtp_forward_level` (`dsv4.rs:5303`): `h' = e_proj(enorm(emb)) + h_proj(hnorm(h_prev))`
→ one full layer (reads the **frozen target layer's** committed KV ring,
`mtp_frozen_target_layer_idx`) → head + `mtp_topk_device`. Chain `draft_chain`
(`spec_decode.rs:582`): `depth = --mtp-draft-tokens` clamped `[1,8]`, default 2;
topk default 1 (only widens candidate matching, adds no verify rows).

### 6.2 Chain verify (one FlashMLA sparse call per layer)
`SpecVerifySchedule` (`dsv4.rs:405`) = `positions[r]=start_pos+depth` +
`ancestors[r]` (parent chain, self-excluded; depth-2 → `[[],[0],[0,1]]`).
`Dsv4ChainVerifyAttnMeta` uploads `[n,max_anc]` -1-padded; kernel
`arle_chain_verify_build_indices_kernel` (`csa_prep.cu:216`) lays each index row
as `[committed SW offsets | ancestors+self | compressed part | -1 tail]` →
one sparse pass verifies all `depth+1` rows with exact chain causality,
**without writing the slot's rolling SW cache**.

### 6.3 Accept + commit-fold
`accept_path` (`spec_decode.rs:190`): a row extends the path only if the verify
argmax both is in that row's top-k *and* equals the drafted token; first mismatch
stops, target argmax becomes the bonus. `commit_accepted_fold` (`dsv4.rs:1806`)
**re-ingests the persisted attn-normed rows** (each layer's `normed` was D2D-copied
into `slot.spec_normed[layer]` during verify, `dsv4.rs:4814`) rather than re-running
a full forward, then `flashmla_alloc_append(m)` + `seq_len=start_pos+m`. Rejected
rows are never gathered.

### 6.4 Rollback snapshot — full mutated-buffer enumeration
This is the area `AGENTS.md` flags as the hard-won EAGLE bug (`truncate_decode_len`
once restored only `compressed.seq_len`, missing `pending_kv`/`prev_overlap`/
`sw_window`/`fp8_kv_pool` ring slots). Order (`spec_decode.rs:327`):
`truncate_slot(start_pos) → restore_spec_ring_tail → commit_accepted_fold`. Snapshot
`Dsv4SpecRingSnapshot` (`attention.rs:3295`) taken pre-draft, pre-allocated per
(slot,layer), D2D (no per-step alloc).

| # | Device buffer | Writer | Rollback disposition | file:line |
|---|---------------|--------|----------------------|-----------|
| 1 | bf16 SW ring boundary slot | draft (non-frozen) | **snapshot/restore** (only boundary slot) | capture `attention.rs:4045` / restore `:4079` |
| 2 | FP8 KV-pool ring slot (data+scale) | draft FlashMLA ring | **snapshot/restore** (page re-resolved both ends) | `:4047 / :4081` |
| 3 | `flash.fp8_kv_comp_packed_rows` | draft | **snapshot/restore** | `:4043 / :4084` |
| 4 | `flash.fp8_kv_sw_bootstrapped` | draft | **snapshot/restore** (else next decode skips the repack) | `:4044 / :4089` |
| 5 | compressor `pending_kv/pending_score/prev_overlap_*` | `dsv4_compressor_update_*` | **frozen-gate self-heal** — verify skips compressor (`:11799`); draft is SW-only | gate `:11799` |
| 6 | compressor/indexer `compressed.seq_len` | length advance | **`truncate_decode_len` rollback** (recompute `total/ratio`) | `:3953 / :11870` |
| 7 | DSA index-K cache `compressed.data` | index-key ring write | **frozen-gate self-heal** (`:11619`; `skip_frozen_compressor` `:9259`) | `:11619 / :9259` |
| 8 | `dsa_official.packed_rows` | `csa_select` advance | **`truncate_decode_len` rollback** (`min(total/ratio)`) | `:3963` |
| 9 | FlashMLA pool append cursor | commit append | **`truncate` rollback** → commit re-advances by `m` | `dsv4.rs:1157 / 1178` |
| 10 | `slot.seq_len` | length | **rollback then re-set** `start_pos → start_pos+accepted` | `dsv4.rs:1174 / 1869` |
| 11 | `slot.spec_normed[layer]` | verify persist | **no rollback** (scratch, fully overwritten each verify) | `dsv4.rs:4814` |
| 12 | verify-local `HiddenStates` | verify scratch | **no rollback** (written before read, freed at return) | `dsv4.rs:3842` |

`restore_spec_ring_tail` asserts the window matches `captured_start_pos/depth` and
`accepted_n ≤ depth` (`:4066`) — a stale snapshot can never be replayed. Whole-slot
LRU swap uses a separate fuller `Dsv4LayerImage` (`Dsv4CompressorImage` +
`Dsv4FlashMlaImage`), distinct from this per-step spec rollback.

**Correctness invariant**: the verify is *pure* (frozen gate skips all committed-KV
writes; chain-verify packs K as a transient chunk), so the only speculative dirty
write is the draft's write to the frozen target layer's boundary SW/FP8 ring slot
— exactly the buffers `Dsv4SpecRingSnapshot` protects — while length/cursor
counters are recomputed by `truncate_decode_len` + `flashmla_truncate_slot`.

### 6.5 Frozen vs commit verifier; gates
`dsv4_verify_frozen` (`attention.rs:75`) is a process-global atomic; while set,
`mla_attention` skips every committed-KV write. **Frozen verifier**
`forward_tokens_verify_scheduled` (`dsv4.rs:2054`) → persistent-scratch
`forward_tokens_verify_stream_persistent` (`dsv4.rs:4351`), writes no slot KV.
**Commit/selftest verifier** `forward_tokens_verify` (`dsv4.rs:1968`) is
non-frozen and writes slot KV — selftest only. **Off by default**:
`spec_decode_on = mtp_draft_tokens.is_some() || dspark_on`;
`--spec-type` defaults `None`; `--spec-type mtp` defaults draft tokens to 2;
CUDA-only. Adaptive skip via `ARLE_DSV4_MTP_ADAPTIVE` (B=1 only).

---

## 7. Latest spec-decode practice — DeepSeek DSpark (2026-06-27)

> **Nothing to train — the trained head is public.**
> `deepseek-ai/DeepSeek-V4-Flash-DSpark` (MIT) ships it as a 3-stage draft chain,
> 4705 tensors matching `Dsv4DsparkStage` field-for-field: `mtp.0.main_proj.
> {weight,scale}` + `main_norm` (entry), `mtp.2.{hc_head_*, confidence_head.proj,
> markov_head.markov_w1/w2}` (exit). `config.json` carries every key
> `merge_dspark_metadata` requires: `dspark_block_size 5`,
> `dspark_target_layer_ids [40,41,42]`, `dspark_markov_rank 256`,
> `dspark_noise_token_id 128799`. Only the ~10 GB draft delta (shards 46-48/48)
> is needed; requantize MXFP4→FP8 with `scripts/requant_dspark_mxfp4_to_fp8.py`.
> DSpark on DSv4-Flash is **shipped and served** (TP=4, all ranks load it).
>
> **The wall is trigger, and capacity is unmeasured.**
> [baselines](baselines.md)' DSv4 DSpark arm reads ~no-op because
> `--dspark-max-prompt-tokens 64` routes multi-k-token prompts to no-spec: not
> measured, not ineffective. On capacity, a 2026-07-14 binary logged an explicit
> `DSpark draft reserve 19000MB → pool_total 141MB, affordable 1`; **that reserve
> term no longer exists.** At HEAD `load_dspark_draft` shards the draft's experts
> and attention through the same `ExpertSplit`/`TpConfig` as the trunk (≈1/EP of
> the ~20 GB fp8 draft per rank), and the only budget term is a small per-slot
> `latent_kv + attn` over `sliding_window + block_size` (`executor/dsv4.rs:560`).
> Read the budget line at HEAD before designing anything.
>
> Paper: *DSpark: Confidence-Scheduled Speculative Decoding with
> Semi-Autoregressive Generation* (DeepSeek × PKU, 2026-06-27); code
> [DeepSpec](https://github.com/deepseek-ai/DeepSpec) (MIT; DSpark + DFlash +
> EAGLE-3). **DeepSpec publishes no V4 config** — only Qwen3-4B/8B/14B and
> Gemma-4-12B — so training a V4 head means writing the DSv4 target adapter
> yourself, plus a hidden-state cache (~38 TB at DeepSpec's Qwen3-4B default).
> Below: authors' claims, hypothesis until reproduced under our bench spec.

### 7.1 The problem DSpark targets
Existing draft families trade off two ways:
- **Autoregressive drafts (EAGLE-3)**: strong token-dependency modeling, high
 accept rate, but draft cost grows linearly with block length → forced to short
 blocks / shallow nets. (This is *our* current MTP family — a sequential nextn
 head, chain-verified.)
- **Parallel drafts (DFlash)**: all draft positions in one forward, cost
 ~independent of block length, but (a) **suffix decay** — independent per-position
 prediction can't model intra-block dependency, so accept rate collapses in the
 block's tail; (b) the optimal verify length is hard to fix, and verifying all
 tokens indiscriminately hurts system throughput under high concurrency.

### 7.2 Mechanism 1 — semi-autoregressive draft
Hybrid: a **parallel backbone** emits hidden states + base logits for all
candidate positions in one pass, then a **lightweight sequential head** injects
prefix dependency token-by-token (cheaply). Two head variants:
- **Markov head**: low-rank projection depending only on the previous token
 (`markov_rank` ~256).
- **RNN head**: a GRU-style cell accumulating the full prefix.

**Draft width = TRUNK width.** The head shares the target's frozen embedding and
LM head, so a narrower backbone would need a lossy projection back up over a
129,280-vocab logit path; no published design does it. V4 head `hidden_size 4096`
(= trunk), `z-lab/Qwen3.6-27B-DFlash` 5120 / 5 layers (= its trunk) — hence
`load_dspark_head`'s `ensure!(cfg.hidden_size == trunk_hidden)`. The paper's
`draft_hidden_size~1024` sketch is not an artifact shape. Size lever is **layer
count**, saturating early (DSpark's 2-layer beats DFlash's 5-layer). V4: 3
stages, `block_size 5`, taps `[40,41,42]`.

### 7.3 Mechanism 2 — confidence-scheduled verification
A **confidence head** predicts a per-draft-token survival probability; a
**hardware-aware scheduler** then dynamically picks the verify length to maximize
*system* throughput (not just per-request accept length). It stops verifying once
predicted survival drops below a throughput-aware threshold — avoiding wasted
target-model verification on low-probability tail tokens under high concurrency.

### 7.4 Claimed results
- Accept-length **+16%–31% over EAGLE-3 and DFlash** on Qwen3 / Gemma4.
- Single-user gen speed **+60%–85% (V4-Flash)**, **+57%–78% (V4-Pro)**;
 high-concurrency throughput **up to +400%**.
- Cross-model: Qwen3 (4B/8B/14B), Gemma4-12B, on top of V4-Flash/V4-Pro.

### 7.5 Mapping onto our runtime (gap analysis, license-or-kill before any code)
| DSpark piece | Our current state | Gap / where it would land |
|--------------|-------------------|---------------------------|
| Sequential MTP head, chain verify | **Have it** — `Dsv4MtpLayer` + chain-verify §6 (the EAGLE-style lane DSpark contrasts) | — |
| Parallel backbone draft | **Have it** — `dsv4/dspark.rs` drafts all `block_size` positions in one pass from the official 3-stage head; verify reuses `forward_tokens_verify_scheduled` | — |
| Markov / RNN sequential head | **Implemented** — `dspark-sp+markov` draft checkpoints load the Markov head; a DSpark train sidecar trains it in-production via acceptance-weighted policy gradient + probability matching (Phase 1 shipped 2026-07-20) | Lightweight head on top of the parallel backbone; small autograd surface (`crates/autograd`) or a fused kernel |
| Confidence **head** | **Have it** — `mtp.2.confidence_head.proj` loads; `dspark_verify_keep` cumprods it into survival and feeds the goodput budget (`qwen35_spec::dspark_verify_lens`) | — |
| Throughput-aware **scheduler** | **Have the core** — `dspark_verify_lens` picks the verify budget maximizing `(R + Σ survival)/step_time(B)` with an additive SPS cost model (`--dspark-sps-*-ms`). Missing: STS calibration (no fitted temperatures yet) and profiled-per-box SPS tables | Remaining pieces are assets (calibration data, per-box SPS profile), not code (#124). |
| Verify length = dynamic | Fixed `depth` (`--mtp-draft-tokens`) | Make `depth` per-step adaptive from a confidence signal + batch occupancy |

**Verdict framing (per `AGENTS.md` §0):** before committing engineering, any
DSpark-style change needs a wall-clock A/B against our current MTP lane on the
SLO workload — not a per-NVTX-window share, and not the paper's single-user
numbers. The confidence-scheduled verify length is the highest-leverage,
lowest-risk first probe (it reuses the existing chain-verify + rollback
machinery; only the `depth` decision changes), and it is exactly the axis where
the runtime already has the batch/SLO state the scheduler needs.

**Acceptance bar = absolute accepted tokens/step, never a % of the window** —
confidence scheduling makes the window per-request, so a ratio bar retightens
itself. Calibration: EAGLE-3-class heads reach τ ≈ 2.5–3.4 on 120B–235B MoE, V3
native MTP 3.96, DSpark on V4-Pro ~5. **Pair it with a wall-clock gate at
production concurrency**: MoE verify overhead runs 1.5–3×, and at high
concurrency break-even acceptance can exceed 1.0 — no acceptance recovers the
loss. Managing that is what the scheduler is for; hence the scheduler, not the
head, is the acceptance lever.

---

## 8. Kernel Roadmap (post-2026-06-30 profile)

Priority follows measured TP=4 B=4 wall-clock/kernels.

| Priority | Work | Status | Gate |
| --- | --- | --- | --- |
| **P0** | Extend DeepGEMM to compressor/indexer batched projections (`compressor.wkv`, `compressor.wgate`, `indexer.weights_proj`; `indexer.wq_b` shares the same helper). | Built and verified on H20 TP=4; no DeepGEMM fallback. HTTP c=4 is noisy, so use phase/nsys for final verdict. | Keep only for `M>1`; M=1 stays scalar. |
| **P1** | Fuse RMSNorm + FP8 activation pack (`dsv4_rms_norm_fp8_quantize`): compute norm and directly encode E4M3 + E8M0 block scale in registers. | Next. Target is the MoE `rms_norm` + `dsv4_deepgemm_pack_quantize_bf16_to_fp8` pair (measured ~12.1%). | A/B on B=4 decode; correctness via needle gate. |
| **P2** | FLUX `ag_gemm` allreduce-GEMM overlap (ByteDance/SGLang practice). | Deferred; integration is multi-day. | Decide after P0+P1 wall-clock A/B. |

---

## Symbol index (stable anchors)

| Concern | Symbol | File |
|---------|--------|------|
| Forward dispatch | `forward_tokens_impl`, `forward_tokens_stream_impl` | `dsv4.rs` |
| Batched decode lane | `forward_decode_batch_stream_impl`, `decode_lane_fwd` | `dsv4.rs`, `attention.rs` |
| MLA attention | `mla_attention`, `mla_attention_prepare`, `mla_attention_fwd` | `attention.rs` |
| FlashMLA decode | `try_flashmla_decode_attention`, `sparse_decode_fwd_batched` | `attention.rs` |
| FlashMLA prefill | `try_flashmla_prefill_attention` | `attention.rs` |
| FP8 KV pack | `flashmla_pack_*`, `dsv4_fp8_kv_pack.cu` | `attention.rs`, `csrc/attention/` |
| DSA indexer | `csa_select`, `csa_select_official`, `dsv4_dsa_official.cu` | `attention.rs`, `csrc/misc/` |
| Paged MQA logits | `dsv4_deepgemm_fp8_paged_mqa_logits_fused_cache_cuda` | `deepgemm_native.cu` |
| MoE routing | `dsv4_route`, `dsv4_route.cu` | `moe.rs`, `csrc/moe/` |
| Grouped GEMM | `deepgemm_grouped_experts*`, `sm90_fp8_gemm_1d2d_impl` | `moe.rs`, `vendor/deepgemm/` |
| DeepEP transport | `dsv4_moe_forward_deepep{,_ll}`, `deepep.rs` | `moe.rs`, `infer-cuda/src/` |
| MTP spec-decode | `Dsv4MtpLayer`, `forward_tokens_verify_scheduled`, `commit_accepted_fold` | `dsv4.rs` |
| Rollback snapshot | `Dsv4SpecRingSnapshot`, `truncate_decode_len`, `restore_spec_ring_tail` | `attention.rs` |
| KV budget | `kv_budget_num_slots`, `dsv4_dsa_key_cache_bytes` | `dsv4.rs`, `attention.rs` |

## References

- Stage-B shared-pool payoff: `experience/wins/2026-06-10-dsv4-lever-gate-license-or-kill.md`
- B=1 graph wash: `experience/errors/2026-06-10-dsv4-wholestep-graph-production-path-wash-rekill.md`
- EAGLE rollback bug: `AGENTS.md` §0.1 (DSv4 EAGLE rollback, 2026-06-06)
- MoE i32 overflow (masked vs contiguous): `experience/errors/2026-06-05-dsv4-prefill-moe-i32-overflow-crash.md`
- DSpark: *Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation*, DeepSeek × PKU, 2026-06-27 (DeepSpec, MIT)
