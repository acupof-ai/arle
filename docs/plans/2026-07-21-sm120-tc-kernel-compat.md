# sm_120 (Blackwell RTX PRO 6000) — ThinkingCap-Qwen3.6-27B-FP8 kernel compatibility map

> Status: Active

Scope: deliverable **(A) correctness-first portable run** on sm_120. **(B)** peak-perf
Blackwell port is enumerated in the appendix only, not planned.

**Verdict — 整齐划一 holds.** The TC 27B FP8 path has **0** ops without a portable
fallback. Getting a correct sm_120 run is **1 code edit** (a dispatch bug) + build
with the right arch list; every Hopper-only kernel already auto-routes to a
portable fallback via runtime `compute_capability()` dispatch. This is "complete +
route the portable tier", not a per-kernel Blackwell rewrite.

## 0. Model resolution — TC-27B is DENSE, not MoE

Ground truth: `crates/infer-vulkan/src/config.rs:9-10` — the on-box `Qwen3.6-27B`
checkpoint is `arch = qwen35` (**dense**); the MoE arch `qwen35moe` is the *35B-A3B*
variant. The loader hard-splits on `general.architecture` (`config.rs:74-77`):
`qwen35` ⇒ `num_experts = 0`. Per-layer type is by tensor presence (`config.rs:277`:
`ssm_conv1d.weight` ⇒ LinearAttention, `attn_q.weight` ⇒ FullAttention), so the 27B
is a **dense hybrid**: GDN linear-attention layers + periodic full-attention layers,
all FFNs dense SwiGLU.

Consequence: the MoE grouped-FP8-GEMM + host-router fallback paths (`qwen35.rs:~4184`,
`moe.rs:539`, `warm_fp8_deepgemm_grouped_prefill` at `qwen35.rs:2719`) are **never
entered** for TC — out of scope for (A).

**Pod-only unknown:** the FP8 *safetensors* `config.json` is not on this box (only
`models/Qwen3.5-0.8B`, `models/Qwen3-0.6B`). Confirm on the pod that the 27B FP8
checkpoint has `num_experts = 0` / no router tensors. The dense conclusion is from
the GGUF sibling + code, not the exact FP8 config.

## 1. Op inventory (TC dense FP8 hybrid: prefill + decode + rollout)

Method: every `csrc` file grepped for `wgmma / tcgen05 / mma.sync / cp.async.bulk /
__nv_fp8 / asm volatile / sm_90a / __CUDA_ARCH__`. Global result — **wgmma**: only a
comment (`gemm/quantized_gemv.cu:98`); **tcgen05**: none in-tree; **TMA**: only
`gemm/deepgemm_native.cu`; **mma.sync**: none; **sm_90a**: only
`attention/arle_fa3_shim.cu`; **__CUDA_ARCH__** guards: `kv/transfer.cu:30`,
`comm/custom_all_reduce.cuh:117/175`, `gemm/quantized_gemv.cu:1281` (`==700` Volta).

| # | op | file:line | Hopper-fast | portable fallback | HW asm? | sm_120 status |
|---|----|----|----|----|----|----|
| 1 | dense FP8 proj GEMM | dispatch `ops/quant_linear.rs:535` | DeepGEMM `deepgemm_native.cu:1641` (sm_90a) | dequant→cuBLAS `quant_linear.rs:441` + GEMV `quantized_gemv.cu:1349` | DeepGEMM=TMA/wgmma; fallback=`__nv_fp8` | **DISPATCH BUG G1** |
| 2 | KV write + q prep | `attention/decode_prep_paged.cu`; `prefill_attention_hd256_prep` | — | in-tree | none | portable |
| 3 | full-attn prefill | `attention/nonpaged_prefill_attention.cu` (`qwen35.rs:6020`) | FA3 shim (`qwen35.rs:5960`) | this row | prefill: none; FA3: sm_90a | portable (FA3 auto-off) |
| 4 | full-attn decode | `nonpaged_prefill_attention_devpos` (`qwen35.rs:5945`) | FA3 split / FA2-sm70 | this row | none | portable |
| 5 | batched decode / reduce | `attention/fused_attention.cu` (`qwen35.rs:7691`) | — | in-tree | none | portable |
| 6 | paged-attn resolve | `fused_attention.cu` `resolve_paged_attn_v1/_fp8_v1` | — | in-tree | none | portable |
| 7 | attention gate hd256 | `attention_gate_batch_hd256` / `_paged_hd256` | — | in-tree | none | portable |
| 8 | FA3 hopper fwd | `attention/arle_fa3_shim.cu` | this (opt-in) | falls to #3/#4 | **sm_90a wgmma** | build-pinned sm_90a, runtime-off |
| 9 | FA2 sm70 | `attention/arle_fa2_sm70.cu` | — | falls to #3/#4 | none | gated off (major<8) |
| 10 | conv1d | `recurrent/conv1d.cu`, `conv1d_decode_batch.cu` | — | in-tree | none | portable |
| 11 | GDR prefill | `recurrent/gated_delta_rule.cu` `_prefill_recurrent` | FlashQLA (sm_90a) | this row | none | portable |
| 12 | GDR decode | `recurrent/gdr_decode_batch.cu`, `gated_delta_rule.cu` `_decode` | — | in-tree | none | portable |
| 13 | FlashQLA chunked GDR | `recurrent/gdr_prefill_*.cu` + TileLang AOT | this (opt-in) | falls to #11 | TileLang sm_90a cubin | runtime-off (flag default false) |
| 14 | gated RMSNorm | `norm/norm.cu` `rms_norm_gated` | — | in-tree | none | portable |
| 15 | l2norm q/k | `norm/norm.cu` | — | in-tree | none | portable |
| 16 | input/post norm | `norm/norm.cu` `rms_norm_offset` | — | in-tree | none | portable |
| 17 | embedding | `elementwise/*` `embedding_batched` | — | in-tree | none | portable |
| 18 | SwiGLU act | `elementwise/elementwise_basic.cu` `silu_mul` | — | in-tree | none | portable |
| 19 | residual add | `elementwise/elementwise_basic.cu` `add`/`add_batch` | — | in-tree | none | portable |
| 20 | dense bf16 GEMM (MLP/o-proj) | `ops.rs:187/340` `gemm_cuda` | cuBLASLt | cuBLASLt | none | portable (cuBLAS Blackwell) |
| 21 | RoPE | folded into prep kernels | — | in-tree | none | portable |
| 22 | sampling | `sampling/*` `argmax_batch`, `sample` | — | in-tree | none | portable |
| 23 | spec-decode | `dspark_draft_sample`, `_chain_accept`, `_filter_probs` | — | in-tree | none | portable |
| 24 | LoRA merge (rollout) | `qwen35.rs:8834` `lora_device_gemm`; `dequantize_fp8_block_scaled_to_bf16` (`:5206`) | cuBLAS + dequant | same | `__nv_fp8` | portable |
| 25 | KV page transfer | `kv/transfer.cu:30` | — | in-tree | generic PTX | portable |

**~25 kernel families (~30 FFI entry points).** Off-path but present (DSv4-only, not
TC): `decode_attention_{quantized,turboquant,varlen_fp8}.cu` (0 hits for
wgmma/tcgen05/sm_90a — portable anyway), `dsv4_*`, `moe/*` grouped, `marlin_*` (int4).

## 2. Gap list

**G1 (the only blocker — FIXED 2026-07-21). Dense FP8 GEMM SM-gate mis-selects
Blackwell.** `quant_linear.rs:171` was `major >= 9`; sm_120 has `major = 12` → gate
returned true → `dsv4_deepgemm_fp8_gemm_nt` launched → `deepgemm_native.cu:1641`
`if (prop.major != 9) return CUDA_ERROR_NOT_SUPPORTED` → `.result()?` hard-aborts the
run. Same defect flows through the warm path (`qwen35.rs:2662`). This is **not** a
missing fallback — the portable dequant→cuBLAS + scalar block-scaled GEMV path
already exists; the gate simply failed to pick it on `major != 9`.
Fix: `major >= 9` → `major == 9` (DeepGEMM is Hopper-exclusive; `deepgemm_native.cu`
checks `== 9` at every entry: 1155/1505/1574/1641/1703). Zero-code interim:
`--qwen35-deepgemm false` (`args.rs:818`).

**Ops with NO portable fallback: 0.** Every other Hopper-only op is build-pinned to
sm_90a (dormant) or runtime-gated off, each with a wired portable fallback:
FA3 `qwen35.rs:782` `== (9,0)`; FlashQLA `runtime_flags.rs:126` default false;
FA2-sm70 `qwen35.rs:831` `major < 8`.

## 3. Build story for sm_120 (T2 opt-in)

- Enable: `TORCH_CUDA_ARCH_LIST="12.0"` (or `"9.0;12.0"` for a fat binary).
  `build.rs:12` lists `120` in `T2_SMS`; `validate_sm`/`parse_sm_token` accept `12.0`→`120`.
- Compiles for sm_120: every `.cu` gets `arch=compute_120,code=sm_120`
  (`build.rs:2825` else-arm) — all TC ops in §1.
- Stays OFF sm_120 (no assemble failure): `build.rs:2816-2823` pins FA3/FlashMLA/shims
  (`is_sm90a_only`) to `compute_90a,sm_90a` regardless of the arch list, so the wgmma
  asm never targets sm_120. FA3/FlashMLA also opt-in (`ARLE_CUDA_ENABLE_FA3`); default
  build swaps stubs.
- `deepgemm_native.cu`: gets the sm_120 gencode but is a host-side JIT harness
  (device kernels are CUTLASS strings JIT'd at runtime for sm_90a); the TU builds for
  sm_120 as it already does for T1 80/86/89. Opt-in via `enable_deepgemm_native`.
- FP8 intrinsics: the dequant fallback uses `__nv_fp8_e4m3`/`__nv_fp8x4_e4m3`
  (`quantized_gemv.cu:50,150,354,619`), native on sm_89+/Blackwell. The only
  arch-conditional (`__CUDA_ARCH__ == 700`, `:1281`) is Volta-only; sm_120 takes the
  portable `#else` (`:1349`).
- No sm_90a-only asm reaches the sm_120 gencode: only `arle_fa3_shim.cu` is sm_90a
  (pinned away); `kv/transfer.cu` asm is generic `ld/st.global` PTX;
  `custom_all_reduce.cuh` is guarded `>= 800`/`>= 700` and retained-out on single-GPU.

## 4. Phased (A) plan

1. **Build** `crates/cuda-kernels` with `TORCH_CUDA_ARCH_LIST="12.0"`; confirm
   FA3/FlashMLA stayed sm_90a-pinned (`build.rs:2816`).
2. **Fix the dispatch defect** — `quant_linear.rs:171` `>= 9` → `== 9` (**DONE
   2026-07-21**). Interim alternative: `--qwen35-deepgemm false`.
3. **FA3 auto-off** (no edit): `qwen35.rs:782` `== (9,0)` excludes sm_120.
4. **FlashQLA GDR off** (no edit): `runtime_flags.rs:126` default false.
5. **FA2-sm70 off** (no edit): `qwen35.rs:831` `major < 8` excludes sm_120.
6. **Gate**: serve the 27B FP8 checkpoint on the sm_120 box and run
   `scripts/needle_gate.py` across `115..8000` (spanning the 241 boundary); end state
   = exact needle recall + `deterministic? true` per length.

**6 steps, 1 code edit (landed); the rest are build + verify-existing-gate.**

## 5. (B) appendix — peak-perf Blackwell port (enumerated only)

A separate, throughput-only effort would add a native sm_120 fast tier: a **tcgen05
FP8 block-scaled GEMM** (CUTLASS 3.x sm_120 collective — 5th-gen tensor cores,
`tcgen05.mma` + tensor-memory, replacing the sm_90a wgmma+TMA JIT in
`deepgemm_native.cu`, wired as a third route in `quant_linear.rs` behind a
`major==12` gate) and a **Blackwell FlashAttention** (FA3/CUTLASS sm_120 build of
`arle_fa3_shim.cu`, un-pinning `build.rs:2816` + an hd256 sm_120 instantiation),
optionally a Blackwell TileLang sm_120 chunked-GDR cubin. None required for (A).
