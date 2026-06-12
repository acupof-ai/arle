# Qwen lane kernel alignment to SGLang — adopt, stop hand-writing

**Directive (ckl, 2026-06-12):** "kernel 全对齐 sglang 吧 不要再自己写了" —
align the Qwen3.5/3.6 CUDA lane's kernels with SGLang's kernel choices
wholesale; new hand-written kernels require explicit approval. Strengthens
the standing adopt-official-first rule from "prefer adopt" to "adopt by
default; the burden of proof falls on keeping a hand kernel".

**Evidence basis:** at HEAD the 35B B=1 decode runs 93.5 tok/s ≈ 14% of the
~670 tok/s weight-bytes roofline, eager (whole-step graph opt-in OFF in the
measured run). SGLang/vLLM's stock stack lands ~35-40% roofline on the same
model class (vLLM FP8 on desktop Blackwell: 208.6 tok/s single-stream) with
zero bespoke kernels and no megakernel: decode CUDA graph default-on,
per-layer fused ops, native-MTP spec decode, FP8/W4 weights. The pod has a
dev SGLang at `/workspace/sglang` (editable) with first-class
`qwen3_5_moe` support (`qwen3_5.py`, `qwen3_5_mtp.py` is_nextn) plus
`sgl-kernel` 0.3.21 and `flashinfer` + `flashinfer_cubin` wheels — the
reference implementation AND the lift sources are already on the box.
The Qwen3.6-35B-A3B checkpoint carries a native MTP head
(`mtp_num_hidden_layers=1`, 19 `mtp.*` tensors).

## Hard constraints

- **No Python / no torch on the hot path** (repo charter). Adoption
  mechanisms, in order of preference:
  1. **Triton AOT**: compile SGLang's Triton kernels offline
     (`triton.compile` → cubin + launch metadata), shapes baked per model
     config — the exact pattern of the existing TileLang AOT lane
     (`crates/cuda-kernels/build.rs` + `csrc/tilelang_dispatch.c`,
     `INFER_TILELANG_PYTHON`); add `INFER_TRITON_PYTHON` alongside.
  2. **Vendored CUDA + C FFI**: lift `.cu` sources (sgl-kernel csrc,
     causal-conv1d) into `vendor/` with raw-pointer C shims, stripping
     torch::Tensor wrappers — FlashMLA/DeepGEMM precedent.
  3. **Prebuilt cubins**: `flashinfer_cubin` artifacts + thin launcher.
- **License discipline unchanged**: every swap ships behind the same-harness
  A/B + needle gate; a losing SGLang kernel stays opt-in with the verdict
  recorded (precedent: DeepGEMM-masked decode lost to warp-per-row at R=8,
  06-11 adoption survey). Default direction is adopt.
- **No half-states**: hand kernels stay as the baseline arm until the
  SGLang counterpart is licensed; removal is a separate deletion commit.

## Mapping — decode step, ours → SGLang

| Op (decode) | Ours today | SGLang counterpart | Mechanism | Tranche |
|---|---|---|---|---|
| GDR recurrent scan (30 layers) | hand `gated_delta_rule_decode_cuda` (csrc/misc/gated_delta_rule.cu) | `fla/fused_recurrent.py` fused_recurrent_gated_delta_rule | Triton AOT | T1 |
| conv1d decode (30 layers) | hand `conv1d_prefill_cuda` seq=1 branch (csrc/misc/conv1d.cu) | causal_conv1d_update (Dao kernel, vendored by sgl) | vendor + FFI | T1 |
| GDR projections | 4× cuBLASLt GEMV per layer | `gdn_fused_proj` jit triton (fused proj post-process) | Triton AOT | T1 |
| gated RMSNorm | hand `rms_norm_gated_cuda` | `fla/layernorm_gated` | Triton AOT | T1 |
| MoE expert FFN (40 layers) | hand grouped decode pair `moe_bf16_grouped_gemm_swiglu_decode` + `_decode` (infer-cuda/src/moe.rs:806/857) | `fused_moe_triton` (gate_up GEMM + silu-fused + down GEMM, all 256 experts grouped) | Triton AOT; **A/B at R=8 mandatory** — our warp-per-row beat tensor-core tiles here before | T2 |
| router topk/renorm/count/scan/pack/scatter/combine (7 launches) | hand dsv4_route + 6 csrc/moe kernels | sgl-kernel topk_softmax + moe_align_block_size + triton epilogue reduce | lift + Triton AOT | T2 |
| full-attn paged decode hd256 (10 layers) | hand `nonpaged_prefill_attention_devpos_cuda` + `hd256_prep` | flashinfer paged GQA decode (hd256), prebuilt cubins on pod | cubin + launcher | T3 |
| q/k-norm + rope + KV append | hand `hd256_prep.cu` | flashinfer fused rope/append path | with T3 | T3 |
| spec decode | none on Qwen lane | `qwen3_5_mtp.py` NEXTN — checkpoint-native MTP head | reuse ARLE DSv4 MTP infra, not a kernel lift | T4 |
| decode CUDA graph | opt-in `ARLE_QWEN35_DECODE_GRAPH` | default-on (`disable_cuda_graph=False`) | flag-default flip after T-gates | T0 evidence → flip license rides each tranche |
| sampling | argmax + host sample | sgl-kernel device top-k/top-p | lift | T5 |
| norms/residual adds | hand, separate launches | sgl fused add_rmsnorm | lift | T5 |
| dense GEMMs (qkv/o/lm_head) | cuBLASLt | torch.mm → cublas (same class) | no change | — |

## Sequencing

- **T0 (evidence, queued as session task #19, next GPU window):** graph-flip
  A/B + nsys per-kernel GB/s + **SGLang same-box bs=1 baseline ± MTP arm** on
  the same checkpoint/GPU. T0's SGLang number is the alignment target; its
  per-op attribution orders T1-T3 by actual ms, and pre-empts megakernel
  speculation (if stock SGLang ≥2.5× us, the gap is adoption, not research).
- **T1 GDN lane** (30 of 40 layers, biggest layer count): fused_recurrent +
  causal_conv1d + gdn_fused_proj + gated-norm. Builds the Triton-AOT lane
  that T2 reuses.
- **T2 MoE lane**: fused_moe_triton + routing epilogue. A/B vs decode-band
  hand kernels at R=8 decides the default; both outcomes recorded.
- **T3 full-attn lane**: flashinfer paged decode hd256.
- **T4 MTP**: checkpoint-native head through the engine spec-decode path
  (frozen-KV semantics per the DSv4 MTP work).
- **T5 small-kernel parity + graph default flip** once capture-safety holds
  across the adopted set.

Each tranche: own commit(s) + bench entry + needle gate; per-tranche
kill-or-license recorded in wins/errors. Megakernel work is explicitly
**off the table** unless T0 shows stock-SGLang parity still leaves ≥2×
on the floor (then revisit per-layer persistent kernels as a separate
licensed experiment).

## Non-goals

- DSv4 lane: untouched (FlashMLA/DeepGEMM/DeepEP already vendored-official).
- Prefill kernels: separate pass after decode parity; FA3/FlashQLA prefill
  paths keep their current gates.
- Metal lane: out of scope (MLX already upstream-aligned).
