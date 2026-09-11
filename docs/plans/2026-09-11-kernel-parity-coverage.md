# CUDA kernel parity-coverage audit

Date: 2026-09-11 · Status: parity gates code-complete for the flagged production families;
this section is the live inventory. Read-only census; paths are repo-relative.

**Status (updated 2026-09-11).** Code-side gates exist for **10/10** of the
production-kernel families this audit originally found without a numeric gate
("Production kernel gate status" below; each row links the merged PR and the
gate file). "10/10" is at family granularity — the table marks in-family
remnants that stay ungated (DSA pack/store index kernels, DeepGEMM synthetic
weights). Geometry mismatches: **2 open of 4 tracked** — the other two are
closed (one in code pending the GPU batch, one ruled not-a-gap), see
"Geometry mismatches" below. **CUDA parity GPU batch runs: 0** — the Vulkan
device gates below have been run on MoltenVK (see "Vulkan"), but every CUDA
parity example/test in the CUDA sections is GPU-pending and runs together
through `scripts/parity_gpu_batch.sh` (positive + `--negative-control` per
family), not by ad-hoc invocation.

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
| `gdr_decode_cuda` / `gdr_decode_batch_cuda` | **HOT** (c=1 default; c≥2 batch) | **P** `gdr_decode_parity` (B=1/8, (16,48)/(8,24), 5 carried steps, output+state vs f32 anchor) | — |
| `conv1d_decode_batch_cuda` | HOT (c≥2) | **P** same `gdr_decode_parity` (conv output + shifted ring) | — |
| `*_prefill_recurrent_varlen_cuda` + `conv1d_prefill_varlen_cuda` (attn_tp≥2 spec verify / DSpark replay) | WARM | **P** `gdr_varlen_parity` (lengths 1/2/5/17/64, B=1/4/8, nonzero init, output+ring+final state vs f64) + FlashQLA-vs-varlen cross-path check at (16,48) lengths 5/17/64 with max-diff output | — |
| `gated_delta_rule_prefill_recurrent_cuda` (non-chunked / fq-unavailable) | WARM | **U+P** training-side vs CPU; `gdr_varlen_parity` covers the same kernel math via the varlen ABI at the serve head set | single-row host ABI itself only driven in autograd tests |
| `conv1d_prefill_cuda` | WARM | **U+P** (training) + **P** (`gdr_varlen_parity` FlashQLA cross-check drives the single-row prefill ABI) | — |
| 5 AOT `gated_delta_rule_chunk_{cumsum,a,recompute,state,o}` (serving build matrix) | **DEAD** | **—** zero callers + no test | kernels.toml rows with no symbol consumer anywhere (subsequently deleted) |
| `batched_copy_uniform_cuda` | HOT (c≥2 setup) | E2E only | memcpy, low risk |

### 2. Full attention — FA3 / FlashMLA / FA2 / paged
| Kernel | Heat | Gate | Geometry gap |
|---|---|---|---|
| `arle_fa3_fwd_hd256_bf16_cuda` (vendored FA3) | **HOT** H20 | **P** `fa3_hd256_shim_parity` (decode split-KV/PackGQA/combine + varlen prefill, #323) | GPU run pending |
| `arle_fa3_fwd_hd256_quant_cuda` (FP8/INT8 paged split-KV) | HOT under quant pool | **P** same `fa3_hd256_shim_parity` (production 1-byte FP8/INT8 pools, incl qlen 256/kv 65537, #323) | GPU run pending |
| `paged_attention_v1` TileLang AOT (25-row resolve) | HOT sm80 / fallback on H20 | **—** (no test/example launches any AOT paged row) | AOT table has hd256 (q8/16/24 kv1/2/4) + one hd64 row, no hd128 — investigated 2026-09-11: hd128 is unreachable in serving (all CUDA targets hd256; see geometry-mismatch note), no row needed |
| `arle_fa2_sm70_attention_cuda` | COLD (V100) | **U** (`fa2_sm70_matches_host_reference`, dense seq4 q2/kv1 hd256) | tiny dense only; no paged/decode-batch |
| FlashMLA sparse decode/prefill shims (`arle_flashmla_sm90_sparse_{decode,prefill}_fwd`, `get_meta`,`sched_meta`) | **HOT** DSv4 | **P** `flashmla_sparse_decode_parity` (CSA, #320) + `flashmla_hca_decode_parity` (HCA, #324), B=8/boundary positions vs host | GPU run pending |
| FlashMLA CSA/HCA chain-verify build_indices, csa_pack_kv | WARM DSv4 | **—** direct; E2E | prefill chain-verify path not in the sparse-decode gates |
| CP ring (`ring_block_fwd_merge{,_fa3}`, `ring_prefill_*`, `cross_cp_merge`) | COLD (attn_cp≥2) | **U+P** (`device_ring_two_blocks…_gqa_hd128`, q4/kv2; host softmax) + multi-GPU train transport examples | best-covered attention family; CP production heads not the tiny test shape |
| paged prep / decode_prep / gates hd256 (`*_prep_cuda`, `attention_gate_*`) | HOT/WARM | **—** direct; E2E | — |

### 3. DSv4 DSpark / DSA / spec
| Kernel | Heat | Gate | Gap |
|---|---|---|---|
| `dsv4_dspark_draft_attention_cuda` | WARM (spec) | **P** `dspark_dsa_parity` (DSv4-Flash draft latent attention at 512 head_dim, kv up to 4096, #313); E2E counters remain | GPU run pending |
| `nonpaged_prefill_attention_ring_varlen_cuda` / `..._batched_cuda` (Qwen3.8 DSpark drafter) | WARM (spec; batched is the c≥2 path, c=8 acceptance 0%) | **P** `dspark_draft_attn_parity` (40q/8kv hd128 block 7, f64 ring-window oracle, B=1/8; production window-null request-length cap 32775 full attention over unequal long kv_len up to 4096 with no wrap, `--dspark-block-size` clamp block 4, synthetic cap12/window6 wrap-contract; single-vs-batched exact agreement) | real draft activations (random inputs), GPU run pod-only |
| DSA official indexer family (`dsa_fused_q_indexer…`, `hadamard128`, `store_index_k_cache`, `pack_index_row`, …) | WARM | **P** for the fused Q indexer: `dspark_dsa_parity` (partial RoPE + H128 + quant scales, top-k sets vs f32/f64, #313). `store_index_k_cache` / `pack_index_row` packing stays **—** (shape guards only) | partial; GPU run pending |
| MLA q/k prep, o-rope inverse, compressor update, FP8 kv pack, build_indices (single+batched) | **HOT** DSv4 | **—** direct; dsv4_parity gates only the first prefill token | large HOT surface, token-level-only E2E |

### 4. MoE / routing / GEMM
| Kernel | Heat | Gate | Gap |
|---|---|---|---|
| Marlin FP8/W8A16/W4A16/NVFP4 GEMM + GEMV (dense quant) | **HOT** | **P** strong: `marlin_{fp8,w8a16,fp4}_parity/correctness` examples at production shapes vs f64/f32 + 9 GPU unit tests + `kernel_ab_fp4` | best-covered family |
| `dsv4_fp8_grouped_{swiglu,down}_decode` (hand-grouped) | **HOT** 1×H20 FP8 | **P** micro test (`dsv4_fp8_grouped_decode_matches_reference`, hidden/inter=16) + `dsv4_decode_moe_parity` at production geometry (K=4096,N=2048, top-6 fan-out B=1/8 = up to 48 experts over ids 0..=255, every output row vs f64) | prefill grouped MoE still uncovered |
| DeepGEMM native masked/contiguous grouped FP8 + BF16 (`m_grouped_*`) | HOT decode / WARM prefill | **P** `deepgemm_grouped_prefill_parity` at production n=4096 vs f64 per-block fp8 oracle (masked+contiguous, full 256-expert distribution); decode micro U + decode parity above | synthetic weights; swiglu/routing end-to-end only |
| routing family (`dsv4_route`, renorm, count, scan, pack, scatter/combine) | **HOT** | **P** `moe_routing_parity` (f64 spec oracle, learned-bias + hash, #314) | GPU run pending |
| MegaMoE `sm90_*` (vendored) | COLD (TP>1 ∪ transport flag) | **—** | multi-card only |
| W4A8 CUTLASS grouped `w4a8_*_sm90` | HOT on W4A8 ckpt | marlin_fp4 + repack-quality script (partly) | distinct CUTLASS path |
| `nvfp4_to_w4afp8`, expert ptr tables, preflight | COLD load/init | **—** | one-time/host |
| cuBLAS `gemm_{bf16,bf16_f32}` dense proj | HOT/WARM | **U** one sm80-89 dequant+gemm test | production sm90 cuBLASLt config not directly gated |

### 5. Norm / elementwise / sampling
| Kernel | Heat | Gate | Gap |
|---|---|---|---|
| `rms_norm_{batched,_offset,_gated}`, silu_mul(_fused), split2, split_qkv, add, embedding_batched | **HOT** | **P** `elementwise_parity` (rms_norm single/batched, silu split/fused, split, embedding vs f64/f32, #312) | GPU run pending |
| `argmax_cuda` / `argmax_batch_cuda` | **HOT** greedy | **P** `argmax_parity` (kernel contract: lowest-index tie, NaN/−inf, vocab tail) + device-neutral `raw_argmax_gate.rs` (#308) | GPU run pending |
| `dspark_draft_sample` / `dspark_filter_probs` / `dspark_chain_accept` | WARM spec | **P** `dspark_sampler_parity` (filter/draw/chain-accept vs f64, vocab to production width, #312) | GPU run pending |

### 6. KV quant
| Kernel | Heat | Gate | Gap |
|---|---|---|---|
| `quantize_paged_kv_fp8_cuda` / single | HOT under `--kv-cache-dtype fp8/int8` | **U** e4m3 roundtrip over discontinuous pages (`fp8_paged_kv_quantize_roundtrip_discontinuous_pages`, #310) + the older INT8 roundtrip (2 heads/hd64/32 tok) | GPU run pending |
| `paged_attention_quantized_fa3_workspace_bytes` | host | n/a | — |
| `quantize_kv_bf16_to_int8` / `dequantize_kv_int8_to_bf16` safe wrappers | test-only | INT8 roundtrip unit calls them (`gemm_tests.rs`) | retained — that test is the only INT8-KV gate |

## Production kernel gate status

All ten families the initial audit flagged as E2E-only now have a code-side
numeric gate. Every row is **Gated (GPU run pending)**: the gate compiles and
is wired into `scripts/parity_gpu_batch.sh`, but no on-device run is recorded
yet (header status).

| # | Kernel family | Default-path role | Gate file | PR |
|---|---|---|---|---|
| 1 | FlashMLA sparse decode + metadata shims (CSA ratio 4, plus HCA ratio 128) | HOT DSv4-Flash decode | `crates/infer-cuda/examples/flashmla_sparse_decode_parity.rs` (CSA, B=8 batched); `crates/infer-cuda/examples/flashmla_hca_decode_parity.rs` (HCA, explicit B=1) | #320, #324 |
| 2 | FA3 hd256 paged shims (`arle_fa3_fwd_hd256_{bf16,quant}_cuda`) | HOT H20 quant-pool decode + bf16 prefill | `crates/infer-cuda/examples/fa3_hd256_shim_parity.rs` (bf16 decode/varlen prefill; FP8/INT8 1-byte pools; qlen 256/kv 65537 long case) | #323 |
| 3 | `gdr_decode_cuda` / `gdr_decode_batch_cuda` + `conv1d_decode_batch_cuda` | HOT GDN decode (c=1 and c≥2 batch) | `crates/infer-cuda/examples/gdr_decode_parity.rs` | #308. Attn_tp≥2 varlen prefill replay is separately gated by `crates/infer-cuda/examples/gdr_varlen_parity.rs` (#322); the varlen kernels are deleted by draft #300 once the GPU batch confirms the chunked-FlashQLA path at every TP. |
| 4 | `argmax_cuda` / `argmax_batch_cuda` | HOT every greedy token | `crates/infer-cuda/examples/argmax_parity.rs` (lowest-index tie-break, NaN/−inf contract) + device-neutral `crates/infer-plan/tests/raw_argmax_gate.rs` | #308 |
| 5 | DSv4 routing family (`dsv4_route`, renorm, count/scan/pack/scatter/combine) | many launches per MoE layer | `crates/infer-cuda/examples/moe_routing_parity.rs` (f64 oracle from `deepseek-spec::v4`, both learned-bias and hash ABIs) | #314 |
| 6 | elementwise layer kernels (`rms_norm`, `silu_mul`, split qkv, embedding) | HOT every layer | `crates/infer-cuda/examples/elementwise_parity.rs` (per-family corruption) | #312 |
| 7 | DeepGEMM grouped prefill MoE | WARM large prefill | `crates/infer-cuda/examples/deepgemm_grouped_prefill_parity.rs` (production n=4096, masked + contiguous, per-block f64 FP8 oracle, realistic numerics) | #316. Remaining: weights are still synthetic, and swiglu/routing reduction stays end-to-end. |
| 8 | FP8 paged-KV quantize | HOT under the production FP8 dtype | GPU unit `fp8_paged_kv_quantize_roundtrip_discontinuous_pages` in `crates/cuda-kernels/src/ffi/gemm_tests.rs` (e4m3 over discontinuous pages) | #310. INT8 keeps its older round-trip in the same file. |
| 9 | DSpark draft attention + DSA indexer family | the Qwen3.8 drafter | DSA indexer + DSv4-Flash draft attention: `crates/infer-cuda/examples/dspark_dsa_parity.rs` (#313); the Qwen3.8 ring-varlen drafter attention, single + c≥2 batched: `crates/infer-cuda/examples/dspark_draft_attn_parity.rs` (#331) | #313, #331 |
| 10 | DSpark sampling accept/filter/draft kernels | WARM spec decode | `crates/infer-cuda/examples/dspark_sampler_parity.rs` (filter/sample/chain-accept vs f64) | #312 |

### Geometry mismatches (gate exists, production shape in question)

Status per item; two remain **Open**.

- **attn_tp≥2 GDN — closed in code, GPU run pending.** Chunked FlashQLA is now
  selected at every supported shard: (8,24) attn_tp=2 and (4,12) attn_tp=4
  (#327), (2,6) attn_tp=8 (#329). `crates/infer-cuda/examples/gdr_varlen_parity.rs`
  (#322) cross-checks FQ vs the recurrent varlen kernel at each geometry
  (lengths 5/17/64, output+ring+final state vs an f64 anchor). With all four
  shards routed to chunked FQ, varlen has no production multi-row caller; draft
  #300 deletes it after the GPU batch confirms the gates. FlashQLA AOT also
  compiles (16,8)/(12,4)/(24,8), used only by training backward.
- **paged_attn_v1 hd128 AOT rows — closed as not-a-gap, no row needed.**
  `paged_attn_v1_raw` has one caller (the Qwen3.5-family full attention layer,
  `qwen35_attention.rs:961`); every supported CUDA target is hd256 per
  host/HF config.json (Qwen3.5-0.8B 8/2/256, Qwen3.5-4B 16/4/256,
  Qwen3.6-27B 24/4/256, Qwen3.6-35B-A3B 16/2/256, Qwen3.8-27B 24/4/256), and
  Qwen3 dense (hd128) was removed from CUDA (support-matrix.md). The hd128
  checkpoints on the host are the DFlash/DSpark drafters (32/8/128, 40/8/128),
  which run their own draft attention (`qwen35/dspark.rs`), not paged_attn_v1.
  The hd128 struct at infer-model/src/qwen35.rs is `kv_pool_pages` unit-test
  arguments, not a model fixture. The sm80-89 1-byte fallback for every
  reachable config is gated by
  `crates/infer-cuda/examples/paged_quant_attn_parity.rs` (#330).
- **FA2 sm70 gate — Open.** The host gate is dense seq4/q2-kv1/hd256 only
  (`fa2_sm70_matches_host_reference`); production V100 paged decode and larger
  batches remain uncovered. Tracked as the last audit gap for that backend.
- **DSv4 TP/EP sharding o-projection — Open.** `dsv4_parity` runs TP=8 but
  gates only the first prefill token; the decode-band TP Q-repack is bit-exact
  gated at TP=2/4/8 by `crates/infer-cuda/examples/dsv4_decode_moe_parity.rs`,
  while the o-slice and grouped o-projection are still exercised only through
  that one argmax (dense-BF16 o-proj is cuBLAS). No code gate yet.

## Metal (MLX) hand-written kernel parity

Audit of `fast::metal_kernel` call sites in `crates/mlx-sys/` on 2026-09-11
(CUDA audit #306 never covered Metal; vendored MLX itself stays
vendor-trusted). The actual census is **6 kernel families** (not 8): 4 in
`mlx_bridge.cpp`, 2 in `mlx_qwen35_model.cpp` — the model TU's GDR forward
and tape variants are the only custom kernels there. All 6 are gated by
`crates/mlx-sys/tests/kernel_parity.rs` against an f64 host oracle at
Qwen3.6-35B-A3B-4bit geometry, with a per-family negative control. Inputs
are quantized through bf16 in the oracle so the residual measures kernel
arithmetic. Tests serialize via `mlx_guard()` (parallel Metal JIT races and
aborts).

| Kernel | Computes | Default-path caller | Geometry in gate |
|---|---|---|---|
| `tape_replay` (bridge) | GDR rollback: `state = g*state + k·δ_tape`, replay of recorded innovation tape when sampled draft tokens are rejected | GDR tape-mode rollback in the compiled forward (record enabled during draft; replay via `mlx_tape_replay`) | B=1, T=16, Hk=16, Hv=32, Dk=Dv=128, f32 state |
| `gated_delta_step` (model TU) | GDR linear-attention recurrent step: decay state by g, `δ=β(v−state·k)`, `state+=kδ`, `y=state·q` | 32 of 40 Qwen3.6 layers (`linear_attention`); both decode S=1 and prefill S=16+ | B∈{1,4}, T∈{1,16}, Hk=16 Hv=32 Dk=Dv=128, bf16 q/k/v, f32 g/β/state |
| `gated_delta_step_tape` (model TU) | Same kernel plus the bf16 innovation tape output | Same layers with tape recording on (draft verification) | B=1, T=16, tape checked against bf16-rounded f64 oracle |
| `batched_sdpa_2pass_partials` + `batched_sdpa_2pass_reduce` (bridge) | 2-pass causal SDPA fixed at 16 queries: 128 partial softmax blocks + online reduction | 8 `full_attention` layers, only on the verify batch (`is_verify && seq_len==16`, head_dim 256, GQA 8); cached int8 KV is dequantized to bf16 before the same call | B∈{1,4}, Hq=16 Hk=2 D=256, q_len=16, N∈{16,100,129,1000,4113}, one int8-KV (8-bit/group-128) dequant case |
| `verify_qmm_mma2big_*` (bridge, one source, group_size 32/64/128) | Fixed-M=16 simdgroup-matrix 4-bit affine quantized matmul: `y = x·(nib·scale+bias)ᵀ` | `QWeight::apply(prefer_verify_m16)` for every attention projection on the verify batch; group 64 is the checkpoint quantization | M=16, K=2048, N∈{512,2048}, bits=4 affine, **BF16 scales/biases (checkpoint dtype)**, all 3 group sizes |

Gate structure (`cargo test -p mlx-sys --release`, also a metal-ci lane):
- `gated_delta_step_parity` — y and final state vs f64 oracle, T=1 and T=16.
- `gated_delta_batched_rows` — B=4 with an independent state per row, pins
  the kernel's `B*Hv` z-grid addressing.
- `gated_delta_innovation_tape` — recorded tape vs bf16-rounded oracle.
- `tape_replay_reconstructs_state` — replay kernel vs an f64 replay oracle
  consuming the same bf16 tape.
- `batched_sdpa_2pass_causal` — output vs f64 softmax with the kernel's
  packed-chunk mask (`n ≤ N−16+qi`) at N 16/100/129/1000/4113, B=1 and B=4,
  including K/V rounded through the production 8-bit KV quantizer.
- `verify_qmm_mma2big_group_sizes` — output vs f64 dequant matmul,
  group_size 32/64/128.
- `negative_controls` — one corruption per family (k spike, v spike, tape
  byte flip, value spike, scale spike), each asserted to clear the clean
  error by >10×; measured clean error is ~1e-2 (relative RMS) and corrupt
  error 2.3–342 for every family.

Tolerances are ~3× the measured clean error. The f32 state bound is 3e-6 at
both T=1 and T=16: the kernel keeps state in f32, so the only error source
is bf16 input rounding and it does not accumulate with T. The tape bound
(1.2e-2) reflects the kernel's bf16 store of the innovation itself.

Gate gaps: synthetic random inputs rather than real checkpoint activations.
Production Metal never runs either kernel with B>1 — every compiled-model
FFI entry pins `current_batch_size = 1` (e.g. `qwen35_compiled_step_session`,
`qwen35_compiled_verify_block_summary`; DFlash batched decode loops slots in
`MetalExecutor` and calls step_session per row) — so B=4 is a kernel
addressing guard, not a production shape. QMM covers bits=4 affine only —
mxfp4 mode=1 deliberately routes to stock MLX; the model forward itself
stays behind the needle gate (these tests exercise the kernels at the op
boundary, not a full step).

## DEAD production-shaped wrappers / AOT rows (resolved by deletion)

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

## Vulkan

Date: 2026-09-11 · Backend: `infer-vulkan` + `crates/vulkan-kernels` (llama.cpp
`vulkan-shaders` @ d2462f8f adapted). The gates below are real-device
`cargo test` on MoltenVK 1.4.0 (Apple M4 Pro, Vulkan 1.2, subgroup 32) with
`ARLE_REQUIRE_VULKAN_DEVICE=1`; the serving target is AMD Radeon 8060S
(gfx1151, Strix Halo), which exposes the same Vulkan compute path.

### Served model and its geometry

`docs/support-matrix.md`: the coherent forward is **Qwen3.5 27B dense
`Qwen3.6-27B-Q8_0` GGUF** (Qwen3.5 GDN architecture), plus the Qwen3.6
35B-A3B MoE FFN. Production dimensions used below:

| Surface | Dense 27B |
|---|---|
| hidden / FFN inter | 5120 / 17408 |
| full-attn heads / head_dim | hd **256**, **24 query / 4 KV heads** (GQA group 6); `attention.key_length` from GGUF (`config.rs:101`) |
| rotary | partial: `rope.dimension_count = 64` of the 256-dim head (`config.rs:124-137`) |
| linear attn channels | 16 key heads + 48 value heads × 128 = **10240** |
| conv kernel | K=4 |
| MoE (35B-A3B, same lane) | 256 experts, top-8; hidden **2048**, expert intermediate **512**; gate/up contract K=2048→512, down K=512→2048 |

Weight quants reachable on the serving path (`forward.rs::gemv_id_kernel_for`,
the dense `record_gemv` projections):
**Q4_K, Q5_K, Q6_K, Q8_0**, activations quantized on-device to Q8_1.

### Coverage matrix — every compute shader vs the serving path

Gate legend: **P** = on-device test vs host f32/f64 oracle · **P(geo)** =
tested AT production geometry · **P(toy)** = oracle exists but only at a small
synthetic shape.

#### Quantized matmul (HOT every decode)
| Shader (Kernel) | Computes | Serving caller | Existing test | Production-geometry? |
|---|---|---|---|---|
| `mul_mat_vecq_q4_k` (GemvQ4K) | Q4_K weight × Q8_1 activation dot per row | `forward.rs` dense proj | **P(geo)** `device_production_geometry` (oracle: `infer-gguf` spec dequant) | K=5120 and 17408, per-family corruption gate |
| `mul_mat_vecq_q5_k` (GemvQ5K) | Q5_K variant | `forward.rs` dense proj | **P(geo)** `device_production_geometry` | K=5120 and 17408 |
| `mul_mat_vecq_q6_k` (GemvQ6K) | Q6_K variant | `forward.rs` dense proj | **P(geo)** `device_production_geometry` | K=5120 and 17408 |
| `mul_mat_vecq_q8_0` (GemvQ8_0) | Q8_0 variant | `forward.rs` dense proj | **P(geo)** `device_production_geometry` | K=5120 and 17408 |
| `mul_mat_vec_id_{q4_k,q5_k,q6_k,q8_0}` (GemvIdQ*) | fused top-k expert GEMV in one dispatch | `forward.rs::record_gemv_id` MoE FFN (gate/up ne11=1, down ne11=top_k) | **P(geo)** `device_production_geometry` (independent dequant·dot oracle per selected id) | **35B-A3B shape**: 256 experts/top-8, gate/up K=2048→512, down K=512→2048; Q4_K/Q5_K/Q6_K/Q8_0; 8 output rows sampled across the full width per expert; sparse 256-slot table proves id dereference |
| `q8_1_quantize` (QuantizeQ8_1) | f32 activation → Q8_1 x4 blocks | `forward.rs` before every GEMV | **P(geo)** `device_production_geometry` (block d + i8 spec-exact at both widths) | K=5120 and linear 10240 |

#### Elementwise / norm (HOT)
| Shader | Computes | Serving caller | Existing test | Production-geometry? |
|---|---|---|---|---|
| `rms_norm` | x·rsqrt(mean x²+eps) | `forward.rs` | **P(geo)** `device_elementwise` (n∈{256,**5120**,**17408**}) | yes |
| `swiglu` | SiLU(gate)·up | `forward.rs`, `model_qwen36` | **P(geo)** `device_elementwise` ({256,**17408**}) | yes |
| `add` | residual add | `forward.rs` | **P(geo)** `device_elementwise` ({256,**5120**}) | yes |
| `scaled_add` | acc + s·x | **not called** (MoE accum uses qwen36_moe_weighted_accum) | **P(toy)** `device_elementwise` | n/a — off serving path |
| `sigmoid_mul` | σ(gate)·val, in-place | `forward.rs` attn/linear gate | **P(geo)** `device_elementwise` ({256,**5120**}) | yes |
| `f16_kv_pack` | f32 K/V row → f16 RNE | `forward.rs:1320` kv pack (once per K/V head row, hd=256) | **P(geo)** `device_elementwise` (bit-exact RNE, n∈{**256**,1024}) | yes — 256 is the served head row, 1024 a kv_dim-wide block |
| `geglu` | GELU(gate)·up | **not called** (model uses SwiGLU) | — | off serving path |
| `swiglu_clamped` | clamped SiLU gating | compiled, **not called** by infer-vulkan | — | off serving path |

#### Full attention (HOT dense 27B)
| Shader | Computes | Serving caller | Existing test | Production-geometry? |
|---|---|---|---|---|
| `rope_neox` | partial/full Neox rotary | `forward.rs:1164` (in-place, slots 0 and 3) | **P(geo)** `device_full_attention`: hd256/rotary256 full-row and **hd256/rotary64 partial** in-place, the same buffer bound in both slots | yes |
| `flash_attn` | online-softmax SDPA, f16 K/V | `forward.rs` full-attn block (GQA mapped host-side: `kvh = qh / 6`, one plane per KV head, gqa_ratio=1) | **P(geo)** `device_full_attention`: hd256 single-head kv∈{1,2,8,33,64,65,200} and **hd256 GQA 24q/4kv** kv∈{1,9,128,257,4096}, oracle reads K/V at f16 precision | yes |
| per-head `rms_norm` | head-wise norm inside attn / GDR | `forward.rs:1278/1297` q/k norm hd256; `:1684` GDR ssm_norm n=128 | **P(geo)** `device_full_attention`: n=**256** q/k and n=**128** GDR value-head | yes |

#### Linear attention GDN (HOT dense 27B)
| Shader | Computes | Serving caller | Existing test | Production-geometry? |
|---|---|---|---|---|
| `qwen35_ssm_conv` | K=4 depthwise causal conv + SiLU + ring advance | `forward.rs` linear block | **P(geo)** `device_linear_attention` (nc∈{7,**10240**}, seq 1/5) | yes |
| `qwen35_gated_delta_net` | gated-delta recurrent state update | `forward.rs` linear block | **P(geo)** `device_linear_attention` (nk16/nv48/hd128, seq 1/2, nonzero state) | yes |

#### MoE routing (HOT 35B-A3B)
| Shader | Computes | Serving caller | Existing test | Production-geometry? |
|---|---|---|---|---|
| `qwen36_router_topk` | softmax → top-k → renorm | `model_qwen36.rs` | **P(geo)** `device_router_topk` (**256** experts top-8 norm/non-norm) | yes |
| `qwen36_router_gemv` | F32 router/shared-gate GEMV | `model_qwen36.rs` | **P(geo)** `device_router_topk` (256×**2048**, n_out=1 sigmoid) | yes |
| `qwen36_moe_weighted_accum` | Σ_e weight_e·expert_e | `model_qwen36.rs` | **P(geo)** `device_router_topk` (hidden **2048**, count 8/1) | yes |

#### Off the Vulkan serving path (no gate required here)
`rope_norm`, `silu`, `get_rows`, `soft_max`, `argmax` (elemental llama.cpp
shaders retained for parity with the borrowed corpus; the forward uses
`rope_neox`, `swiglu`, fused kernels and CUDA-side samplers). All 9
`dsv4_*` shaders + `qwen35_gated_delta_net` DSv4 variants are compiled for
the DSv4 bring-up but `infer-vulkan` serves Qwen3.5/3.6 only — zero
infer-vulkan callers; their gates belong to the CUDA DSv4 audit above.

### Vulkan gaps closed by these gates

Production-geometry parity tests added (host oracles independent of the
shaders; quant formats decoded from the GGUF spec via `infer-gguf`):

1. **GEMV Q4_K/Q5_K/Q6_K/Q8_0 at 5120 and 17408 contraction**, sampled output
   rows (the oracle is f32 dequant·dot over the SAME block bytes), with a
   per-family corruption assertion.
2. **Q8_1 quantize oracle at 5120/10240**: block d/s + i8 checked against an
   f32 host quantizer transcribed from the Q8_1 spec.
3. **fused GemvId Q4_K/Q5_K/Q6_K/Q8_0** vs an INDEPENDENT per-expert
   dequant·dot oracle (the existing `device_gemv_id` cross-checks against the
   plain device GEMV) at the served 35B-A3B routing shape: 256 experts/top-8,
   gate/up K=2048→512, down K=512→2048. The 256-slot weight table is sparse
   (only the 8 routed slots populated, ids scattered …,255) so the gate proves
   `data_ids[slot]` dereference; 8 output rows per expert are sampled across the
   full width. Numeric assertions are TWO-ARMED: rel-L2 plus an elementwise
   slope+floor bound `|g-w| <= 0.03|w| + 0.03·rms(want)` (the floor keeps the
   bound finite for near-zero outputs; worst element printed). Each family has
   its own 3×-expectation corruption control that must trip both arms.
4. **f16_kv_pack at n=256** (the served full-attention head row; n=1024 covers
   a kv_dim block), bit-exact host RNE.
5. **rope_neox hd256/rotary64** partial in-place (slots 0 and 3 aliased),
   **flash_attn hd256 at the served GQA 24q/4kv** with kv lengths
   {1,9,128,257,4096}, and **per-head rms_norm n=256** (full-attn q/k) with a
   second n=128 case for the GDR `ssm_norm` caller.

### Device availability gate

Device tests silently skip when no Vulkan device exists, so a green CI without
an ICD proves nothing. Set `ARLE_REQUIRE_VULKAN_DEVICE=1` to turn
"no device" into a panic (`tests/common/mod.rs::require_device`, shared by
every device test file). Verification command on this Mac (MoltenVK):

```
ARLE_REQUIRE_VULKAN_DEVICE=1 \
VK_ICD_FILENAMES=/path/to/molten_icd.json \
DYLD_LIBRARY_PATH=<dir with libvulkan.dylib> \
cargo test -p vulkan-kernels -p infer-vulkan --features vulkan
```

Without the env var the same binary prints `skipping device test` and passes;
with it set and no ICD it panics `ARLE_REQUIRE_VULKAN_DEVICE set but no Vulkan
device is available: …` and exits non-zero.

Already at production geometry and unchanged: rms_norm/swiglu/add/sigmoid_mul,
ssm_conv, gated_delta_net, and the three qwen36 router kernels.
