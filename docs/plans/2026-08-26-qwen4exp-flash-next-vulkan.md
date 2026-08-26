# Qwen3.8-Flash-Next (`qwen4_exp`) on the Vulkan lane

Status: active. S0 shipped (`a971fbcb1`). Target box: Radeon 8060S (gfx1151),
74.43 GiB device-local heap, 63.6 GB OS-visible RAM, 256 GB/s LPDDR5X.

Reference implementations, both read rather than inferred from:
- `transformers` 5.16 `models/qwen4_exp/modeling_qwen4_exp.py` (2707 lines) —
  authoritative for semantics. Its predecessor `qwen3_next` is what ARLE's
  existing `qwen35` path implements, so diffing the two isolates the new work.
- SGLang [#36497](https://github.com/sgl-project/sglang/pull/36497), branch
  `sgl-project:qwen4-main-squashed`, 95 files, +18543/−84 — authoritative for
  what a serving implementation actually has to build. Not merged.

## Verdict

It runs, at **Config B**: non-expert weights requantized to Q8_0, MTP and the
vision tower kept, the PLE/n-gram table file-backed outside device memory.

| | device GiB | free | decode (projected) |
| --- | ---: | ---: | ---: |
| A — non-expert as-shipped BF16 | 71.35 | 3.08 | ~12.2 tok/s |
| **B — non-expert Q8_0, +vision +MTP** | **69.17** | **5.26** | **~20.3 tok/s** |
| C — non-expert Q4_K, +vision +MTP | 66.81 | 7.62 | ~31.5 tok/s |

Requantizing the non-expert half is the whole game: **87% of decode bytes are
the BF16 non-expert weights** (8.62 GB/token) and only 1.33 GB is routed
experts. It also pays for MTP and vision twice over.

MTP is worth keeping precisely because this box is bandwidth-bound: batched
verify amortizes one weight sweep across N tokens. It is 4.86 GiB only because
the vendor scoped quantization to routed experts; requantized like everything
else it is 1.41 GiB. It is gated on batched MoE forward — run sequentially,
verify costs 2× the base forward and speculative decode is a net loss at any
acceptance rate.

Measured composition of the 125.91 GiB checkpoint (safetensors headers):

| bucket | GiB | placement |
| --- | ---: | --- |
| routed experts (NVFP4, 192 shards) | 63.32 | device, packed |
| PLE n-gram table (FP8, 10 shards) | 47.68 | **file-backed, never device** |
| MTP draft layer (own 512-expert MoE) | 4.86 | device (requantized 1.41) |
| linear_attn (36 layers) | 3.89 | device |
| lm_head / embed_tokens | 1.18 / 1.18 | device / host gather |
| hyper-connections (97 GatedResidual) | 1.19 | device |
| self_attn + indexer (12 layers) | 1.15 | device |
| vision ViT (27 layers) | 0.84 | device |

## Precision: W4A16

`ggml`'s `block_nvfp4` (`ggml-common.h:211`) is **the same primitive** as
modelopt's `group_size: 16` NVFP4 — E2M1 nibbles with one UE4M3 scale per 16
values, ggml grouping four of them into a 64-value block. Conversion is a
repack, not a requantize.

Which activation width is available is decided by the vendored shaders:

| path | NVFP4? | gives |
| --- | :---: | --- |
| `dequant_funcs.glsl` → `mul_mat_vec.comp` (f16 acts) | yes | **W4A16 decode** |
| `mul_mm_funcs.glsl` (coopmat GEMM, f16 acts) | yes | **W4A16 prefill** |
| `mul_mat_vecq_funcs.glsl` (q8_1 acts) | no | W4A8 decode — needs writing |
| `mul_mmq_funcs.glsl` (q8_1 GEMM) | no | W4A8 prefill — needs writing |

So W4A16 is free on both lanes and W4A8 is not. Take W4A16: the checkpoint was
calibrated at W4A4 (static activation scales), and feeding wider activations can
only improve on that — the static scales simply go unused. The bandwidth cost is
nil, since activations are 5 KB against ~9 GB of weights per token.

**Converter constraint, hard.** modelopt carries an extra per-tensor F32
`weight_scale_2` that `block_nvfp4` has no slot for, and both ways of absorbing
it are blocked: widening the block scale to FP16 costs +7.03 GiB (experts
63.32 → 70.31) and overflows the heap; re-encoding it into the E4M3 block byte
puts 99.98% of groups below E4M3's min normal 2⁻⁶, for 10.8% mean / 34.3% max
error on the scale — silently. It must ride out-of-band as a per-expert F32.

This is also why the loader reads **safetensors directly** rather than going
through a GGUF converter: six tensor families here (hyper-connections, indexer,
n-gram shards, PLE, MTP, vision) have no ggml naming convention, and the scale
plane would have to be invented too — two sides of a private contract nobody
else validates, versus reading the names the reference implementation defines.

## The PLE/n-gram table is a gather, and that decides everything

`Qwen4ExpTextNGramEmbedding` (`modeling:1018`) hashes token n-grams with
splitmix64 multipliers into 16 heads (`(ngram_size−1) × heads_per_ngram`), each
with a prime vocab just above 20M, and finishes with `self.ngram_embedding(ids)`
— an `nn.Embedding` lookup. Per token that is **16 rows × 160 FP8 = 2560 bytes**.
51.2 G params that decode almost never reads.

Two measurements settle the placement (`device_ple_mmap_probe`, opt-in):

- **A 60 GiB device-local allocation moved OS-visible free RAM by 4.2 GiB**
  (48.4 → 44.2). The GPU carve-out and the host heap are largely disjoint, so
  the OS keeps its page cache for the table. Earlier planning assumed the device
  allocation would eat host RAM; it does not.
- The gather costs **1.518 ms/token at one thread and 0.011 ms at 32** — a 138×
  fan-out effect. A cold-cache variant over freshly-written files read 0.585 ms
  at QD16. Either bound is under 1% of an ~82 ms step. **Serializing it is the
  only way to get this wrong**, and `memmap2` exposes no `PrefetchVirtualMemory`
  on Windows, so the fan-out must be explicit.

SGLang reached the same conclusion by a different route: `ple_offload_embedding`
puts the table in pinned host memory and gathers on a dedicated stream, one step
ahead (`model_qwen4_exp.py:775-812, 1130-1145`), and their open issue
[#36514](https://github.com/sgl-project/sglang/issues/36514) asks to extend the
offload target to an auxiliary GPU. On this box neither host RAM (37.22 GiB
Vulkan-visible) nor device memory can hold 47.68 GiB, so file-backed is not an
optimization here, it is the only option.

## The four novel components

**Hyper-connections** — the structural one. The inter-layer residual is
`hc_count × hidden = 10240`, stream-major, seeded by `inputs_embeds.repeat(1,1,4)`.
Each `GatedResidual`:

```
hn  = grouped_rmsnorm(h, group=2560)     # 4 independent norms over a 10240 weight
u   = silu(W_down @ hn / 4)              # 10240 -> 320   (note /hc_count BEFORE silu)
m   = sigmoid(W_up @ u)                  # 320 -> 10240
x   = mean_s(m[s] * hn[s])               # -> 2560, the block input
inj = 2 * sigmoid(W_inject @ hn / 4)     # -> 4, in (0,2)
```

and the layer rebuilds `h[s] += inj[s] * y`. Applied twice per layer. **There
are no `input_layernorm` / `post_attention_layernorm` / `model.norm` tensors in
the checkpoint at all** — `hc_norm` is the normalization, and the final
`hyper_connection_mixer` (a `GatedResidual` with `use_combine=False`) collapses
10240 → 2560 and *is* the final norm feeding lm_head.

1.28 GB/token — 13% of the active set, nearly as much as all routed experts
combined — so it is the top quantization target after the experts. It also adds
~500–700 dispatches/token to a decode path already measured as dispatch-bound
(34% of wall). SGLang's `hc_combine.cuh` shows the fusion boundary worth
copying: fuse the **injection half** into one kernel (phase 1 computes the four
`inj` dot products in fp32 with a CTA reduction, phase 2 streams
`out[c·H+i] = residual[c·H+i] + inj[c]·block_out[i]`), and leave the input-mix
half as norm + two GEMVs + a gated mean.

**PLE** sits at `layer_idx = 1` (`ple_layer_ids [2]` is one-indexed) and its
output is added into the hidden state **unconditionally** — omitting it is a
wrong forward, not a degraded one. Bilinear gate with a sign-preserving sqrt,
then a depthwise conv with **kernel 4 and dilation 3** (9-step ring). Two traps:
the residual branch adds the *un-normed* gated value while only the conv branch
takes the normed one, and `sign(0) = 0` must survive.

**QSA indexer** — deferrable, and provably so. `block_topk = 2048/4 = 512`, so
attention is **exactly dense for ≤2051 visible tokens** and only sparsifies at
2052. Shipping short-context first is a correct product, not a compromise. When
it does land: the vendored `flash_attn.comp` masks but still scans all KV, so a
selection mask buys correctness and zero speed — a gathering FA variant is a
separate, later cost.

**Cache contract** gains a third conv-state slot holding **2 int64 token ids**
(the n-gram history), initialized to `eos_token_id`. That is a genuine dtype
break in the "conv state" abstraction, and omitting it means every decode step
after the first hashes against EOS padding: wrong logits, no crash.

## Reuse that was verified, and the traps that look like reuse

Verified by adversarial check, not by inspection:

- **Gated-delta linear attention is an exact match**, shapes *and* split.
  `qwen4_exp` uses separate `in_proj_qkv/z/a/b` — the form ARLE already consumes
  — where `qwen3_next` used a fused, per-k-head-interleaved `in_proj_qkvz`
  needing `fix_query_key_value_ordering`.
- **The gated q_proj `[query|gate]` layout is the shape already in production**
  for the on-box 27B, not an analogous one. `sigmoid_mul.comp` binds gate and
  value in the correct orientation (the highest-risk silent-inversion site).
- **Interleaved M-RoPE is bit-identical to `rope_neox` for text**, since equal
  t/h/w position rows make the sector expression collapse to the same one.
- MoE router formula, shared-expert gate placement and add order, and the
  three-tensor expert split (the NVFP4 checkpoint stores experts *un-fused and
  per-expert*, unlike the BF16 repo's fused `gate_up_proj`).
- `Qwen35Config::from_model_dir` was *run* against the local checkpoint and
  returns every field correctly.

The traps — each produces coherent output with no crash:

1. **`qwen36_router_topk.comp` drops experts 256–511.** `BLOCK 256`, one lane per
   expert; the top-k renormalization hides the wrong softmax denominator.
   Guarded in S0; the strided fix is S1.
2. **The linear-attention output gate is sigmoid, not silu.**
   `output_gate_type: "sigmoid"` vs `qwen3_next`'s hardcoded silu. ARLE computes
   `silu(z) * normed`. The fix is a call swap to `record_sigmoid_mul`.
3. `rms_norm_params_rows` has no `ngroups` — the hyper-connection norm is 4
   groups of 2560 within a 10240 row.
4. GQA head-map needs an interleave flag on non-GGUF loads.
5. NeoX RoPE's `in == out` aliasing is load-bearing, undocumented, unguarded.
6. `RawConfig` is `#[serde(untagged)]`, so any field error surfaces as "data did
   not match any variant".

Three of these sit in code the survey first classified as clean reuse. **The
per-layer parity harness against a dumped torch reference is the highest-value
artifact in this plan** — it is the only thing that turns "reviewed carefully"
into "measured". Build it before chasing the first token, not after.

## Sequence

| step | content | size |
| --- | --- | ---: |
| S0 ✅ | fail-loud guards: arch variant, router bail, device budget cap | done |
| S1 | strided router (512), NVFP4 dtype, FP8 E4M3 dequant | 8 h |
| S2 | safetensors source, slab uploader, mmap'd N-table host embedding | 30 h |
| S3 | NVFP4 fused-expert GEMV + CPU oracle | 14 h |
| S4 | hyper-connection scaffold + **per-layer parity harness** | 40 h |
| S5 | PLE + n-gram → **first token** (indexer stubbed, ≤2048 ctx) | 24 h |
| S6 | QSA indexer → contexts >2051 | 40 h |
| S7 | requantize the non-expert 87%; batched MoE prefill; fuse hyper-connections | 40 h |
| S8 | MTP speculative decode (needs S7's batched verify) | 20 h |

Hours are CC-execution, not calendar.

## Risks

**The device budget is 93% committed at Config B.** Several plausible
implementation choices break it, and the residency cap added in S0 is what turns
that from a plan-level assumption into a load-time refusal.

**Decode is already dispatch-bound** (~5400 dispatches/token, 34% of wall) and
this model adds ~700. Not fatal alone, but as S7 cuts bytes 1.6–2.6× the fixed
dispatch floor becomes a proportionally larger share — so the Q8_0/Q4_K
throughput projections above are bandwidth numbers that ignore it and must be
re-derived against a measured floor before S7's ordering is fixed.

**The indexer could land as a pure slowdown.** Measure masked-dense flash at 8K
and 32K before building the gathering variant; if the mask costs more than
sparsity saves, keep the ≤2048 gate.

**W4A16 quality is untested in fact.** gsm8k 97.27% / aime26 98.75% were
measured at W4A4. Theory says wider activations only help; that needs an eval
run after S5, not an assumption.
