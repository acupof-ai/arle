# CUDA kernel parity-coverage audit

Date: 2026-09-11 · Status: audit, followed by deletions (see the
`kernel-parity-s11` PR) and upcoming parity gates (agenda task
`kernel-parity-gates`). Read-only inventory; paths are repo-relative.

Generalizes the varlen lesson (a GDN recurrence kernel that ran in production
with no numeric-parity gate): enumerate every CUDA kernel reached on the
infer-cuda production path, map the gate that covers it, and flag where the
gate does not exercise the production geometry.

## Scale

- **281** extern FFI declarations under `crates/cuda-kernels/src/ffi/`:
  gemm 69 · misc 46 · attention 39 · recurrent 38 · moe 18 · nccl 19 ·
  comm 16 · sampling 8 · elementwise 8 · embedding 8 · kv 5 · norm 5 · quant 2.
  Most of nccl/comm/misc/embedding are transport / mempool / init (excluded).
- **48** TileLang AOT rows in `crates/cuda-kernels/kernels.toml`: 30
  `gate="flashqla"` (6 geometries × 5 phases) + 12 paged-attention rows + 5
  GDR chunk-prefill + 1 (`gdr_bwd` among the flashqla roots).
- **56** `.cu/.c/.cpp` files under `crates/cuda-kernels/csrc/`.
- The matrix covers the **production-reached compute subset** (~120 wrappers);
  zero-caller production-shaped wrappers are listed under DEAD.

Heat (default 1×H20 / sm90a / ThinkingCap-Qwen3.6-27B-FP8 serve): **HOT** every
decode step/layer · **WARM** prefill/spec/verify · **COLD** load/CP/flag/TP-gated.

Gate legend: **P** = numeric parity example/test vs host ref · **AB** =
kernel_ab · **E2E** = lever/needle/sampling/spec HTTP gate only · **U** = GPU
unit test vs ref · **—** = no gate.

## Coverage matrix by subsystem

### 1. Qwen3.5/3.6 GDN linear attention (recurrent + conv1d)
| Kernel | Heat | Gate | Geometry gap |
|---|---|---|---|
| `gdr_fq_prep` + AOT `gdr_fq_{cumsum,kkt,fwd}` (FlashQLA) | WARM | **U+P** (`crates/autograd` `test_linear_attention.rs`, vs CPU f32 incl grads; k8/v32 **and 16/48**, d128, chunk+batched) | tested heads 8/32 & 16/48; serve launcher only emits (16,32)/(16,48); AOT also compiles (24,8)(12,4)(16,8)(16,16) used only by training backward |
| `gdr_decode_cuda` / `gdr_decode_batch_cuda` | **HOT** (c=1 default; c≥2 batch) | **—** (no direct numeric gate; batched only via `needle_concurrent.py` E2E) | HOT decode kernel with no tensor-parity test |
| `conv1d_decode_batch_cuda` | HOT (c≥2) | **—** direct (conv1d boundary covered training-side only) | same |
| `gated_delta_rule_prefill_recurrent_cuda` (non-chunked / fq-unavailable) | WARM | **U+P** training-side vs CPU | serve geometry path (attn_tp≥2 local K=8) not the tested head set |
| `conv1d_prefill_cuda` | WARM | **U+P** (training) | — |
| 5 AOT `gated_delta_rule_chunk_{cumsum,a,recompute,state,o}` (serving build matrix) | **DEAD** | **—** zero callers + no test | kernels.toml rows with no symbol consumer anywhere (deleted in kernel-parity-s11) |
| `batched_copy_uniform_cuda` | HOT (c≥2 setup) | E2E only | memcpy, low risk |

### 2. Full attention — FA3 / FlashMLA / FA2 / paged
| Kernel | Heat | Gate | Geometry gap |
|---|---|---|---|
| `arle_fa3_fwd_hd256_bf16_cuda` (vendored FA3) | **HOT** H20 | **—** direct; E2E only (`lever --kv-cache-dtype fp8` / needle) | HOT kernel, vendor-trusted, no in-repo numeric gate |
| `arle_fa3_fwd_hd256_quant_cuda` (FP8/INT8 paged split-KV) | HOT under quant pool | **—** direct; E2E only | quantized decode path |
| `paged_attention_v1` TileLang AOT (25-row resolve) | HOT sm80 / fallback on H20 | **—** (no test/example launches any AOT paged row) | AOT table has hd256 (q8/16/24 kv1/2/4) + one hd64 row; **no hd128** — non-sm90 hd128 full-attn paged decode has no row and errors |
| `arle_fa2_sm70_attention_cuda` | COLD (V100) | **U** (`fa2_sm70_matches_host_reference`, dense seq4 q2/kv1 hd256) | tiny dense only; no paged/decode-batch |
| FlashMLA sparse decode/prefill shims (`arle_flashmla_sm90_sparse_{decode,prefill}_fwd`, `get_meta`,`sched_meta`) | **HOT** DSv4 | **—** direct; dsv4 E2E prefill-argmax only | HOT DSv4 decode kernel family with no numeric gate |
| FlashMLA CSA/HCA/chain-verify build_indices, csa_pack_kv | WARM DSv4 | **—** direct; E2E | — |
| CP ring (`ring_block_fwd_merge{,_fa3}`, `ring_prefill_*`, `cross_cp_merge`) | COLD (attn_cp≥2) | **U+P** (`device_ring_two_blocks…_gqa_hd128`, q4/kv2; host softmax) + multi-GPU train transport examples | best-covered attention family; CP production heads not the tiny test shape |
| paged prep / decode_prep / gates hd256 (`*_prep_cuda`, `attention_gate_*`) | HOT/WARM | **—** direct; E2E | — |

### 3. DSv4 DSpark / DSA / spec
| Kernel | Heat | Gate | Gap |
|---|---|---|---|
| `dsv4_dspark_draft_attention_cuda` | WARM (spec) | **—** numeric; E2E + `sampling_gate --require-spec` counters | draft attention, no ref |
| DSA official indexer family (`dsa_fused_q_indexer…`, `hadamard128`, `store_index_k_cache`, `pack_index_row`, …) | WARM | **—** (shape guards only) | whole Deep Sparse Attention indexer has no numeric gate |
| MLA q/k prep, o-rope inverse, compressor update, FP8 kv pack, build_indices (single+batched) | **HOT** DSv4 | **—** direct; dsv4_parity gates only the first prefill token | large HOT surface, token-level-only E2E |

### 4. MoE / routing / GEMM
| Kernel | Heat | Gate | Gap |
|---|---|---|---|
| Marlin FP8/W8A16/W4A16/NVFP4 GEMM + GEMV (dense quant) | **HOT** | **P** strong: `marlin_{fp8,w8a16,fp4}_parity/correctness` examples at production shapes vs f64/f32 + 9 GPU unit tests + `kernel_ab_fp4` | best-covered family |
| `dsv4_fp8_grouped_{swiglu,down}_decode` (hand-grouped) | **HOT** 1×H20 FP8 | **P** micro test (`dsv4_fp8_grouped_decode_matches_reference`, hidden/inter=16) + `dsv4_decode_moe_parity` at production geometry (K=4096,N=2048, top-6 fan-out B=1/8 = up to 48 experts over ids 0..=255, every output row vs f64) | prefill grouped MoE still uncovered |
| DeepGEMM native masked/contiguous grouped FP8 + BF16 (`m_grouped_*`) | HOT decode / WARM prefill | **P** `deepgemm_grouped_prefill_parity` at production n=4096 vs f64 per-block fp8 oracle (masked+contiguous, full 256-expert distribution); decode micro U + decode parity above | synthetic weights; swiglu/routing end-to-end only |
| routing family (`dsv4_route`, renorm, count, scan, pack, scatter/combine) | **HOT** | **—** direct; E2E | high-call-count, no isolated gate |
| MegaMoE `sm90_*` (vendored) | COLD (TP>1 ∪ transport flag) | **—** | multi-card only |
| W4A8 CUTLASS grouped `w4a8_*_sm90` | HOT on W4A8 ckpt | marlin_fp4 + repack-quality script (partly) | distinct CUTLASS path |
| `nvfp4_to_w4afp8`, expert ptr tables, preflight | COLD load/init | **—** | one-time/host |
| cuBLAS `gemm_{bf16,bf16_f32}` dense proj | HOT/WARM | **U** one sm80-89 dequant+gemm test | production sm90 cuBLASLt config not directly gated |

### 5. Norm / elementwise / sampling
| Kernel | Heat | Gate | Gap |
|---|---|---|---|
| `rms_norm_{batched,_offset,_gated}`, silu_mul(_fused), split2, split_qkv, add, embedding_batched | **HOT** | **—** direct; E2E | ubiquitous HOT elementwise, no numeric unit |
| `argmax_cuda` / `argmax_batch_cuda` | **HOT** greedy | **—** direct (one training-side lazy-op argmax test, tiny); behavior via sampling/spec E2E | HOT greedy selector, no serve-path CUDA numeric test |
| `dspark_draft_sample` / `dspark_filter_probs` / `dspark_chain_accept` | WARM spec | **—** numeric; Metal `spec_parity.py` token-level + DSv4 selftest argmax identity | CUDA kernels only behavior-gated |

### 6. KV quant
| Kernel | Heat | Gate | Gap |
|---|---|---|---|
| `quantize_paged_kv_fp8_cuda` / single | HOT under `--kv-cache-dtype fp8/int8` | INT8 roundtrip unit (2 heads/hd64/32 tok); FP8 no direct numeric test | E2E-only for the FP8 production dtype |
| `paged_attention_quantized_fa3_workspace_bytes` | host | n/a | — |
| `quantize_kv_bf16_to_int8` / `dequantize_kv_int8_to_bf16` safe wrappers | test-only | INT8 roundtrip unit calls them (`gemm_tests.rs`) | retained — that test is the only INT8-KV gate |

## Production kernels with no numeric gate (or geometry mismatch), ordered by default-path call frequency

1. **FlashMLA sparse decode kernel + metadata shims** — HOT on the default
   DSv4/H20 path, zero direct numeric gate (only first-prefill-token E2E).
2. **FA3 paged quantized + bf16 fwd shims** (`arle_fa3_*_hd256_*`) — HOT on
   H20 (every quant-pool decode + bf16 short prefill); vendor-trusted, no
   in-repo numeric gate, E2E only.
3. **`gdr_decode_cuda` / `gdr_decode_batch_cuda` + `conv1d_decode_batch_cuda`**
   — HOT GDN decode (c=1 default and c≥2 batch); no tensor-parity test. The
   varlen incident's neighbor: the recurrent decode scan is assumed, never
   directly compared. (S12 adds the host-written parity gate.)
4. **`argmax_cuda` / `argmax_batch_cuda`** — HOT every greedy token; no serve
   CUDA numeric test (a wrong tie-break silently changes output). (S12.)
5. **routing family** (`dsv4_route`, renorm, count/scan/pack/scatter/combine)
   — many launches per MoE layer, E2E only.
6. **rms_norm/silu/split/embedding elementwise** — HOT every layer, no unit.
7. **DeepGEMM grouped prefill MoE** — WARM but large; now compared at
   production n=4096 on masked+contiguous paths by
   `deepgemm_grouped_prefill_parity`; remaining gap is synthetic weights and
   swiglu/routing-reduction end-to-end coverage.
8. **FP8 paged-KV quantize** — HOT under the production FP8 dtype, E2E only
   (INT8 has a round-trip, FP8 does not).
9. **DSpark draft attention + DSA indexer family** — WARM spec path, shape
   guards + counters only.
10. **Sampling DSpark accept/filter/draft kernels** — WARM, behavior-only on
    CUDA (Metal parity doesn't exercise these CUDA implementations).

### Geometry mismatches (gate exists but doesn't cover the production shape)
- **attn_tp≥2 GDN**: local K = 16/attn_tp = 8 at TP2; `fq_geometry_supported`
  rejects (needs local K=16), so serve falls to the un-gated recurrent scan.
  FlashQLA AOT compiles (16,8)/(12,4)/(24,8) but the serve launcher never
  references them (training-backward only). Autograd parity uses heads 8/32
  and 16/48 — not the serve TP≥2 shard.
- **paged_attn_v1 AOT has no hd128 row** (hd256 + one hd64 only): on a
  non-sm90 box, hd128 full-attn paged decode/prefill that declines FA3 has no
  fallback kernel. H20 hides it (FA3 native); A100/sm80 + a hd128 model is the
  exposed config.
- **FA2 sm70 gate** is dense seq4/q2-kv1/hd256 only; production V100 paged or
  larger batches untested.
- **DSv4 TP/EP sharding**: dsv4_parity runs TP=8 but gates only the first
  prefill token; the decode-band TP Q-repack is bit-exact gated at TP=2/4/8 by
  `dsv4_decode_moe_parity`, while the o-slice and grouped-o-proj remain
  exercised only through that one argmax (dense-BF16 o-proj is cuBLAS).

## DEAD production-shaped wrappers / AOT rows (resolved in kernel-parity-s11)

**Scope caveat:** the scan that produced the first draft of this list covered
only the `infer-cuda` **serving** call graph. "Is X wired" requires scanning
the whole tree — training (`autograd`) and GPU unit tests are separate call
graphs (the standing rule in the `substrate-audit-grep-full-tree` feedback).
The two false positives below were caught by a full-tree `rg` and are
**retained**; the genuinely dead items were deleted.

Deleted (zero callers tree-wide):
- `attention::dsv4_fp8_kv_pack` safe wrapper + `arle_dsv4_fp8_kv_pack_cuda`
  extern/native wrapper (strided variants supersede). The shared
  `dsv4_fp8_kv_pack_kernel<448>` template is retained — the strided entry still
  instantiates it.
- The 5 `ffi=false` TileLang AOT rows
  `gated_delta_rule_chunk_{cumsum,a,recompute,state,o}` plus their
  `gated_delta_rule.py` factories and `gen_tilelang_aot.py` WrapperSpecs:
  compiled into the bundle but with no FFI symbol and no launcher anywhere.

RETAINED after the full-tree scan (originally mis-listed as dead):
- `recurrent::gdr_prefill_chunk_prepare_raw` — **live**:
  `autograd/src/backend_cuda/linear_attention_forward.rs` calls it in the
  chunkwise GDN training forward (`gdr_chunkwise_prefill`, default on).
- `kv_quant::quantize_kv` / `dequantize_kv` — **live**: called by the
  `int8_kv_quantize_dequantize_roundtrip` GPU unit test, the only INT8-KV
  numeric round-trip gate.

Intentionally RETAINED (they have a real caller):
`quant_linear::gemv_fp4_e2m1_group` (a `kernel_bench` example calls it) and
the `ring_block_*_{bwd,finalize}` / `ring_block_bwd_fa3` family + ring host
helpers (autograd/training callers).


## Note on the registry
`operators/registry.toml` `correctness_gate = "numeric+e2e"` is a declarative
label, not an executable gate; its consumers are hygiene + an evidence-JSON
reducer. `KERNEL_CAPABILITIES` is a build identity string, not a numeric check.
Neither is parity coverage — the varlen kernel was the same class of
"listed/shipped, never compared".
