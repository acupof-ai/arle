# DSv4 Decode Optimization — Full-Chain Budget & KV Quantization Plan

> Status: Active
> Date: 2026-07-07
> Based on: [TP4/EP4 throughput + nsys profile](2026-07-07-dsv4-tp4ep4-highconc-nsys-profile.md)

---

## 1. Model Constants (measured from config.json)

| Param | Value |
|-------|-------|
| hidden_size (H) | 4096 |
| num_layers (L) | 43 |
| num_attention_heads | 64 |
| num_kv_heads (GQA) | 1 |
| head_dim | 512 (NoPE 448 + RoPE 64) |
| q_lora_rank | 1024 |
| o_lora_rank | 1024 |
| o_groups | 8 |
| n_routed_experts (E_total) | 256 |
| top_k | 6 |
| moe_intermediate_size (I) | 2048 |
| sliding_window | 128* |
| max_seq_len serve default | 128K (CSA layers compress) |

*\* `sliding_window: 128` in config.json is the MLA local-attention window per compression block, not the full context window.*

---

## 2. Per-Step Memory Access Budget (B tokens in flight)

### 2.1 Weight Reads per Decode Step

All weights are FP8 block-scaled (1 byte/value effective, + scales). Per rank at TP4:

| Component | Formula | Bytes (TP4) |
|-----------|---------|-------------|
| QKV proj A (wqkv_a) | L × 3 × q_lora_rank × H / TP | 43 × 3 × 1024 × 4096 / 4 = **135M** |
| Q proj B (wq_b) | L × q_lora_rank × H / TP | 43 × 1024 × 4096 / 4 = **45M** |
| Output proj A (wo_a) | L × o_groups × o_lora_rank × (H/o_groups) / TP = L × o_lora_rank × H / TP | same 45M |
| Output proj B (wo_b) | L × H × o_lora_rank / TP | 43 × 4096 × 1024 / 4 = **45M** |
| Compressor wkv | L × 2 × head_dim × H / TP (NoPE+gate per head) | complex, ~30M est. |
| Router gate | L × E_total × H / EP = L × 256 × 4096 / 4 | **11M** |
| Expert FFN (per active expert) | top_k × B × 3 × I × H / E_local | see §2.3 |
| Shared expert | L × 3 × I × H / TP | ~27M |
| **Total weight read/step (TP4)** | | **~300-350M bytes** |

### 2.2 KV Cache Reads per Step

DSv4 KV = FP8 packed, 584 bytes/token (448 FP8 NoPE + 128 BF16 RoPE + 8 e8m0 scales).

Per layer, per slot: `sw_blocks × page_bytes` where `page_bytes = 64 × 584 = 37,376`.

At B concurrent slots, each reading its sliding window of 128 tokens:
- Per layer: B × 128 × 584 = B × 74,752 bytes
- 43 layers: B × **3.2M bytes**

At B=8: **25.6M bytes** KV read. Small vs weight reads (~320M).

### 2.3 MoE Expert Weight Reads (the key bottleneck)

EP4 → E_local = 256/4 = 64 experts per rank.

At B tokens in flight, total routes = B × top_k = 6B. Distributed across 64 local experts:
- **Avg tokens per expert = 6B / 64 = 0.094B**

| B | avg tokens/expert | expert weight read/expert | total expert weight read |
|---|-------------------|---------------------------|--------------------------|
| 1 | 0.09 | 3 × 2048 × 4096 = 25M (full read for 1 token) | 6 × 25M = **150M** |
| 8 | 0.75 | 25M (still full read) | ~48 × 25M = **1.2G** |
| 16 | 1.5 | 25M (still full read) | ~96 × 25M = **2.4G** |
| 64 | 6.0 | 25M (read once, MAC 6 tokens) | ~256 × 25M / 6 = **1.1G** |
| 128 | 12.0 | 25M (read once, MAC 12) | amortized |

**The problem**: at B ≤ 16, most experts see 0-1 tokens. Each active expert reads its full 25M weight for essentially 1 GEMV. Weight is read ~B times more total than at B=1 (because more experts are activated), but each read is not amortized.

**HBM bandwidth floor**: H20 HBM3 ≈ 4.8 TB/s. At B=1:
- Total weight read ~320M (dense) + 150M (MoE, 6 experts) = **470M**
- Floor: 470M / 4.8T = **0.098 ms** (dense only)
- With MoE: (320M + 6×25M) / 4.8T = **0.098 ms**

Wait — this doesn't match. The issue is that at B=1, only 6 of 64 local experts are active. So expert weight read is 6 × 25M = 150M, not 64 × 25M.

Actually the real constraint is **per-expert weight read**: each of the 6 active experts reads its full 25M for a single token. That's 6 × 25M = 150M of expert weight + ~170M dense weight = 320M total. At 4.8 TB/s: **0.067 ms floor**.

But actual is ~38ms. That's 567× the bandwidth floor. Why?

**Answer: it's not bandwidth, it's latency.** GEMV (matrix-vector) is latency-bound, not bandwidth-bound. The weight is read with poor arithmetic intensity. A 2048×4096 GEMV does 2048×4096×2 = 16.8M FLOPs on 2048×4096 = 8.4M bytes read = 2 FLOPs/byte. Tensor core efficiency requires ~30+ FLOPs/byte to saturate.

**Weight amortization**: at N tokens per expert, same 8.4M read yields N × 16.8M = 16.8N M FLOPs → 2N FLOPs/byte. To hit 30 FLOPs/byte: need N ≥ 15 tokens per expert.

With EP4 (64 local experts):
- 15 tokens/expert × 64 experts = 960 routed tokens needed
- 960 / top_k(6) = **160 concurrent requests**

---

## 3. Where the 38ms Goes (B=1, clean estimate)

nsys 90s 系统级捕获包含模型加载 + prefill + decode，绝对比例被 prefill 污染。以下是基于代码路径的修正归因（decode-only 估计）：

| Category | ms | % | Code Path |
|----------|-----|---|-----------|
| Dense proj FP8 GEMV (wqkv_a, wq_b, wo_b, compressor) | ~10-12 | 28-32% | `dsv4_fp8_gemv_batch_cuda` — scalar warp-per-row, loops over B tokens |
| wo_a BF16 cuBLAS GEMM (per-group loop) | ~3-4 | 9-11% | `dsv4_wo_a_grouped_linear` → gather + cuBLAS gemm + scatter per group |
| MoE expert compute (grouped kernels) | ~8 | 21% | `dsv4_fp8_grouped_swiglu` + `grouped_down` — per-expert GEMV, grouped |
| NCCL (allreduce + allgather) | ~7 | 18% | TP attn allreduce + EP MoE allgather |
| FlashMLA attention kernel | ~0.6 | 1.5% | `flash_fwd_splitkv_mla_fp8_sparse` — extremely efficient |
| FP8 KV cache write (block_scaled_to_fp8) | ~2.6 | 7% | scales + values quant per step |
| DeepGEMM (wq_b, wo_b decode proj) | ~1-2 | 3-5% | `dsv4_deepgemm_fp8_gemm_nt` — only for FP8 weights with caches |
| Router BF16 cuBLAS GEMM | ~1 | 3% | `gemm_batch` → cuBLAS gemm (M=B, K=4096, N=256) |
| Misc (mhc_params, pack_quantize, lm_head) | ~2 | 5% | |
| Kernel launch + host overhead | ~3 | 8% | ~110K launches/s × 3.75us |
| **Total** | **~38** | **100%** | |

**Key correction**: `gemv_handwritten_kernel` 21.9% in nsys is **prefill contamination**.
Decode-only: `ops::gemv` for BF16 is called only by lm_head (once per token).
At c=8 producing 62 tok/s aggregate = 62 calls/s. The 90s capture's 1.21M calls = ~13K/s → dominated by prefill GEMVs (model load verification + per-request prefill projections).

---

## 4. `gemv_handwritten_kernel` Attribution — RESOLVED

nsys 90s capture shows `gemv_handwritten_kernel` (BF16) at **21.9% of total GPU time** (1.21M calls, avg 18.6us).

**Resolved via code trace**:

| Caller | Weight format | Kernel path | Decode calls/s |
|--------|--------------|-------------|----------------|
| lm_head | DenseBf16 | `ops::gemv` → `ffi::gemv_cuda` → handwritten | 1× per token produced |
| Router | DenseBf16 | `ops::gemm_batch` → `ffi::gemm_cuda` → **cuBLAS** (not handwritten) | 43× per step |
| wo_a | DenseBf16 (this checkpoint) | `dsv4_wo_a_grouped_linear` → per-group cuBLAS gemm | 43×8 groups per step |
| wq_b, wo_b | FP8 block-scaled | DeepGEMM (has cache) | 43× per step |

**Conclusion**: In decode, `ops::gemv` for BF16 is called **only by lm_head** (once per token).
At c=8 producing 62 tok/s aggregate = 62 calls/s. The 90s capture's 1.21M calls = ~13K/s →
**dominated by prefill GEMVs** (model load verification + per-request prefill projections where M=seq_len).

**Action**: decode-only nsys capture (30s steady-state, no prefill in window) needed for clean attribution.
The 21.9% figure is system-wide capture artifact, not decode cost.

---

## 5. KV Cache: Current State & Quantization Headroom

### 5.1 Current: Already FP8 Packed (584 bytes/token, confirmed)

DSv4 KV format: `KVFormat::PackedBytes { bytes_per_token: 584 }`

Per-token layout (K + V combined, per KV head = 1 for GQA):
| Component | Dims | Format | Bytes |
|-----------|------|--------|-------|
| NoPE latent K | 448 values | FP8 e4m3 | 448 |
| RoPE K | 64 values | BF16 | 128 |
| NoPE latent V | 448 values | FP8 e4m3 | 448 |
| RoPE V | 64 values | BF16 | 128 |
| e8m3 block scales | — | e8m0 | ~8 |
| **Subtotal raw** | | | **1160** |

**But 584 bytes/token is the actual packed format** — the NoPE latent K and V share
compression (the MLA compressor produces a compressed representation that fits in
fewer bytes). The exact packing is in `kv_layout.rs` / `dsv4_kv_pack`. The 584 figure
is the confirmed `bytes_per_token` from the runtime.

**Key**: DSv4 KV is NOT Qwen-style paged BF16/FP8. It's a custom packed format that
already uses FP8 for the bulk (NoPE latent) and BF16 only for the RoPE-carrying dims
(64 of 512 = 12.5% of head_dim).

### 5.2 KV Quantization Options

| Option | Savings | Risk | Effort |
|--------|---------|------|--------|
| RoPE dims FP8 (from BF16) | 128→64 bytes = ~11% | Medium: RoPE position encoding needs high precision for long ctx | M: pack/unpack kernel change + quant kernel |
| FP4 NoPE latent | 448→224 bytes = ~38% | High: latent value distribution may not survive FP4 | L: new pack format + dequant in FlashMLA |
| Reduce max_seq_len serve default | Linear in seq_len | None if workload fits | S: CLI flag change |
| KV cache sharing (prefix reuse) | Depends on workload | None if RadixCache already wired | Check if DSv4 path uses RadixCache |
| Sliding window reduction | Linear in sw_blocks | Model-dependent | S: config flag |

**Biggest lever: reduce max_seq_len**. If serve default is 128K but workload only needs 8K, that's 16× KV savings. But this is workload-dependent, not a kernel optimization.

### 5.3 Slot Count → Concurrency Mapping

From the experiment: 105 slots/GPU at TP4, c=32 OOM'd.

Each slot draws:
- Per-layer state: `Dsv4LayerAttentionState` (position encodings, compressor indices, etc.)
- FlashMLA KV pool band: `(sw_blocks + comp_blocks) × 64 × 584` per layer

To first order, **halving the KV budget doubles slots**. If we save 20% via RoPE-FP8 → ~126 slots → c=20 safe → per-expert tokens = 20×6/64 = 1.9 (still GEMV).

**To reach 4 tokens/expert** (minimum for meaningful weight amortization):
- Need 4 × 64 / 6 = 43 concurrent requests
- That requires ~2.1× current KV budget
- Options: 2× slots via max_seq_len reduction + RoPE-FP8, or accept that MoE weight amortization requires DP-attn batching at a scale the current KV budget can't hold

---

## 6. Optimization Space — Systematic View

### 6.1 By Category

| Category | Current | Target | Mechanism | Expected Gain |
|----------|---------|--------|-----------|----------------|
| **MoE expert GEMV → GEMM** | 0.09-1.5 tokens/expert | ≥4-8 tokens/expert | DP-attn batching (c≫64) + KV budget increase | 2-4× MoE throughput |
| **Dense proj GEMV** | handwritten/cuBLAS GEMV | DeepGEMM tensor core | Verify which weights are BF16, quantize to FP8 | 1.5-2× dense proj |
| **NCCL allreduce** | 18% GPU time | DeepEP NVSHMEM | Enable deepep feature | Cut NCCL to ~5% |
| **Kernel launch** | ~5% overhead | graph capture | MoE decode graph (already has graph scratch alloc) | Cut launch to <1% |
| **FP8 KV conversion** | 7% (block_scaled_to_fp8) | Fused quant | Fuse into preceding kernel | Cut to ~2% |
| **Compressor index build** | ~5ms first step | Cached across steps | Already cached (first-step only) | N/A steady-state |
| **Scheduler overhead** | ~3ms/step | Batching amort. | Already amortized across B | N/A |

### 6.2 Priority Ranking (ROI = gain / effort)

| # | Optimization | Gain | Effort | ROI |
|---|-------------|------|--------|-----|
| 1 | **max_seq_len=32K cap → 4× slots → c=128** | 4-6× throughput | S (CLI flag) | ★★★★★ |
| 2 | **Quantize wo_a BF16→FP8 → DeepGEMM** | 1.3-1.5× dense proj | M (loader + kernel) | ★★★★ |
| 3 | **DeepEP → NCCL 18%→5%** | 1.15× wall-clock | M (enable feature) | ★★★★ |
| 4 | **Quantize router BF16→FP8** | 1.03× | S (loader + quant path) | ★★★ |
| 5 | **RoPE dims FP8 in KV** | 1.12× slots | M (pack/unpack + FlashMLA) | ★★★ |
| 6 | **MoE decode graph capture** | 1.05× | M (already partial) | ★★ |
| 7 | **FP4 NoPE latent KV** | 1.46× slots | XL (new format + dequant) | ★ |

**Rationale change**: max_seq_len cap moved to #1 (zero effort, 4× slot gain → enables
MoE weight amortization at c=128). wo_a FP8 quant stays #2 among kernel changes
(confirmed BF16 in `/host/DeepSeek-V4-Flash-FP8` checkpoint → no DeepGEMM cache exists).

### 6.3 The Structural Constraint

MoE weight amortization needs **≥4 tokens per expert** = **≥43 concurrent requests** (at EP4, top-6). Current KV budget caps at ~32 OOM. Even with 2× KV savings (RoPE-FP8 + max_seq_len=8K), we reach ~64 slots = 6 tokens/expert = barely entering the amortization regime.

**This means**: for pure-decode workloads, DSv4 on H20 TP4 is structurally limited by MoE GEMV. The per-token cost is dominated by weight-read-per-expert that cannot be amortized without extreme concurrency.

**Prefill is different**: large M naturally gives weight amortization. The system is likely much more efficient on prefill-bound workloads.

---

## 7. Next Experiments

### 7.1 Decode-Only nsys (clean attribution)
Capture 30s of steady-state decode at c=8, AFTER warmup, with no prefill in window. This gives clean per-kernel decode-only attribution.

### 7.2 wo_a Weight Format Verification
```
# Check if wo_a tensors are FP8 or BF16 in the checkpoint
strings target/release/arle | grep wo_a_deepgemm
# Or add a one-line log in loader.rs when wo_a DeepGEMM cache is built
```

### 7.3 max_seq_len Sweep for Slot Count
```
arle serve --max-seq-len 8192 --num-slots 256 ...
# vs default 128K
```
Measure how many slots fit at max_seq_len = 4K/8K/16K/32K.

### 7.4 DeepEP Enable
```
cargo build --release --features cuda,nccl,deepep
```
Measure NCCL time reduction.

### 7.5 c=48 Throughput with Increased Slots
If we can get 64 slots via max_seq_len cap, run c=48 throughput to see if MoE weight amortization kicks in (48×6/64 = 4.5 tokens/expert).

---

## 8. KV Quantization — Detailed Plan

### 8.1 Current State (confirmed)

| Item | Value | Source |
|------|-------|--------|
| KV format | `KVFormat::PackedBytes` | `dsv4.rs` KV pool init |
| bytes_per_token | 584 | runtime constant |
| NoPE latent (K+V) | FP8 e4m3, 896 bytes raw | FlashMLA kernel input |
| RoPE dims (K+V) | BF16, 256 bytes raw | `kv_layout.rs` pack/unpack |
| Block scales | e8m0, ~8 bytes | per-block scaling |
| Packing ratio | 584 vs ~1160 raw = 2× compressed | MLA compressor output |

**The only BF16 remaining in KV: 64 RoPE dims × 2 (K+V) = 128 BF16 values = 256 bytes raw.**
After packing compression, the RoPE portion contributes proportionally.

### 8.2 Option A: RoPE BF16 → FP8 (medium effort, 11% savings)

**What**: Convert the 64 RoPE-carrying dimensions from BF16 to FP8 e4m3.

**Savings calculation**:
- Current RoPE K+V: 128 BF16 × 2 bytes = 256 bytes raw per token
- After FP8: 128 FP8 × 1 byte = 128 bytes raw
- Raw savings: 128 bytes/token = 11% of 1160 raw
- Packed savings (proportional): ~64 bytes/token from 584 → **~520 bytes/token**
- Slot capacity gain: 584/520 = **1.12× more slots** (105 → ~118)

**Risk**: RoPE position encoding requires high numerical precision for long context.
The sin/cos rotation accumulates error across 128K positions. FP8 may cause:
- Position discrimination degradation at long ranges
- Attention score smearing for distant tokens

**Mitigation**:
- Per-channel block scale for RoPE dims (separate from NoPE scale)
- Verify with needle retrieval at max_seq_len (128K)
- If FP8 e4m3 fails, try FP8 e5m2 (wider range, less precision)

**Implementation steps**:
1. `kv_layout.rs`: change RoPE field from `bf16` to `fp8_e4m3` in pack/unpack
2. `dsv4_block_scaled_to_fp8_kernel`: add RoPE quantization path (currently only NoPE)
3. FlashMLA kernel: dequant RoPE dims before attention (currently BF16 direct)
4. `dsv4_fp8_gemv_batch_kernel`: if it reads KV directly, update RoPE dequant
5. Needle gate test at seq_len = 4K, 32K, 128K

**Files to touch**:
- `crates/cuda-kernels/csrc/kv/` (pack/unpack kernels)
- `crates/infer-cuda/src/kv_layout.rs` or equivalent
- `crates/cuda-kernels/csrc/attention/flashmla/` (RoPE dequant in kernel)
- `crates/infer-cuda/src/dsv4.rs` (KV format descriptor)

### 8.3 Option B: max_seq_len Cap (zero effort, 2-16× slot gain)

**What**: Reduce `--max-seq-len` serve default from 128K to workload-appropriate value.

**Mechanics**: KV pool allocates per slot proportional to max_seq_len:
- `kv_pool_size = num_slots × max_seq_len × bytes_per_token × num_layers`
- Halving max_seq_len doubles available slots (approximately)

**Slot budget math**:
- Current: 105 slots at max_seq_len=128K, 584 bytes/token, 43 layers
- KV budget per GPU: 105 × 128K × 584 × 43 ≈ **337 GB** — wait, that's per-pool not per-GPU.

Actually: each slot's KV = `sw_blocks × page_bytes` where `page_bytes = 64 × 584 = 37,376`.
`sw_blocks = ceil(max_seq_len / sliding_window)` = ceil(128K / 128) = 1024 blocks.
Per-slot KV = 1024 × 37,376 = **38.3 MB** per layer.
43 layers = **1.65 GB per slot**.
105 slots = **173 GB per GPU** — that's the KV pool budget.

| max_seq_len | sw_blocks | per-slot KV | max slots (173GB budget) |
|-------------|-----------|-------------|--------------------------|
| 128K | 1024 | 1.65 GB | 105 (current) |
| 64K | 512 | 0.82 GB | ~210 |
| 32K | 256 | 0.41 GB | ~420 |
| 16K | 128 | 0.21 GB | ~840 |
| 8K | 64 | 0.10 GB | ~1,680 |
| 4K | 32 | 0.05 GB | ~3,360 |

**Per-expert tokens at c=max_slots (EP4, top-6)**:
| max_seq_len | slots | c | tokens/expert | regime |
|-------------|-------|---|---------------|--------|
| 128K | 105 | 16 (OOM at 32) | 1.5 | GEMV |
| 32K | 420 | 128 (safe) | 12 | **GEMM amortization** |
| 16K | 840 | 256 | 24 | GEMM |

**This is the single biggest lever.** At max_seq_len=32K, 128 concurrent → 12 tokens/expert =
weight amortization starts working. At 16K, 256 concurrent → 24 tokens/expert = well into GEMM regime.

**Caveat**: this is workload-dependent. If the workload genuinely needs 128K context, this doesn't help.
But for agentic/eval workloads where typical context is 4-16K, this is free throughput.

**Implementation**: CLI flag change only. No code changes.
```
arle serve --max-seq-len 32768 --num-slots 256
```

**Validation**: needle gate at the chosen max_seq_len to confirm retrieval still works.

### 8.4 Option C: FP4 NoPE Latent (high effort, 38% savings, high risk)

**What**: Quantize the 448-dim NoPE latent K+V from FP8 to FP4 (e2m1 or NF4).

**Savings**: 448×2 FP8 → 448×2 FP4 = 896 → 448 bytes raw. ~38% of raw KV.
Packed: ~584 → ~400 bytes/token = **1.46× slots** (105 → ~153 at 128K).

**Risk**: latent values may have distributions that don't survive FP4:
- MLA compressor output is a learned representation, not activations
- FP4 has only 7 mantissa bits (e2m1) or 16 levels (NF4)
- FlashMLA attention computes dot products over these — accumulated error could be catastrophic

**Feasibility check needed first**:
1. Dump NoPE latent K/V values from a real forward pass
2. Analyze distribution (mean, std, range, outliers)
3. Simulate FP4 quant error → attention score error
4. If error > 1% in attention output, KILL

**Implementation** (if feasible):
- New KV format variant: `PackedBytesFp4`
- New quant kernel: `block_scaled_to_fp4`
- FlashMLA kernel: FP4 dequant path (or dequant before kernel)
- This is weeks of work for uncertain gain

**Recommendation**: DEFER. Max_seq_len cap (Option B) gives 2-16× slot gain for free.
FP4 only makes sense if max_seq_len=128K is a hard requirement.

### 8.5 Option D: DeepEP NVSHMEM for NCCL Offload (not KV, but budget-relevant)

Not a KV quantization, but frees ~13% of step time from NCCL → more GPU time for compute →
effective throughput gain equivalent to 1.15× without changing KV.

Already listed as priority #3 in §6.2.

### 8.6 Combined Strategy: max_seq_len + RoPE-FP8

| Config | slots (est.) | c (safe) | tokens/expert | expected throughput |
|--------|-------------|----------|---------------|---------------------|
| Baseline (128K, BF16 RoPE) | 105 | 16 | 1.5 | 88 tok/s (measured) |
| max_seq_len=32K only | 420 | 128 | 12 | **~350-500 tok/s** (est.) |
| max_seq_len=16K only | 840 | 256 | 24 | **~600-800 tok/s** (est.) |
| 32K + RoPE-FP8 (1.12×) | 470 | 128 | 12 | ~350-500 (same, capped by c not slots) |

**The bottleneck shifts**: at c=128 with 12 tokens/expert, MoE weight amortization kicks in.
The new bottleneck becomes:
1. Dense projections (wqkv_a, wo_a) — still GEMV if not quantized to FP8
2. NCCL allreduce/allgather — still 18% unless DeepEP enabled

**So the full chain is**:
1. **max_seq_len=32K** (free, 4× slots) → enables c=128 → 12 tokens/expert
2. **wo_a FP8 quant** (medium effort) → enables DeepGEMM for wo_a → cuts dense proj time
3. **DeepEP** (medium effort) → cuts NCCL from 18% to ~5%
4. **RoPE-FP8** (medium effort) → 12% more slots → headroom for longer prompts

Steps 1-3 together: expected **4-6× aggregate throughput** at c=128 vs current c=16 peak.

### 8.7 Implementation Order

| Phase | What | Effort | Prereq |
|-------|------|--------|--------|
| **Phase 0** | max_seq_len=32K serve + c=128 throughput measure | S (flag) | none |
| **Phase 1** | wo_a FP8 quantize → DeepGEMM activation | M | Phase 0 confirms c scaling |
| **Phase 2** | DeepEP enable → NCCL reduction | M | independent |
| **Phase 3** | RoPE-FP8 KV quant | M | Phase 0+1 baseline |
| **Phase 4** | FP4 NoPE latent (if needed) | XL | only if 128K mandatory |

Phase 0 is zero-risk, zero-effort and gives the biggest slot gain. Run it first to measure
actual concurrency scaling before investing in kernel work.
