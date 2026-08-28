# Qwen3.8-Flash-Next (`qwen4_exp`) on the Vulkan lane

Status: active. S0-S5 shipped — first token out (` Paris`), see the
[win entry](../experience/wins/2026-08-27-qwen4exp-first-token.md). Target box: Radeon 8060S
(gfx1151), 74.43 GiB device-local heap (**70.71 GiB driver budget** -- plan
against the budget, not the size), 63.6 GB OS-visible RAM, 256 GB/s LPDDR5X.

| step | commit | what it bought |
| --- | --- | --- |
| S0 | `5c7cec117` | `qwen4_exp` fails by name instead of loading as a 256-expert MoE; router and device-budget guards |
| S1 | `c059efe35` | router to 8192 experts (strided), NVFP4 + FP8-E4M3 dtypes |
| — | `73b71ba0e` | the cache cliff, measured rather than quoted |
| S2 | `21424b1ab` | safetensors reader, name classifier, slab suballocator — **71.314 GiB residency plan over the real checkpoint, 296,475 tensors, zero unclassified** |
| S3 | `885e221e5` | NVFP4 expert GEMV (repacked planes, plain-f32 B operand) + host oracles for hyper-connections and PLE |
| S4 | `e574b5823` | fused HC/PLE kernels: GatedResidual in 4 dispatches, PLE in 2; device tests 2.1e-6 / 3.6e-7 vs the oracles |

(Hashes are post-rebase onto main and will drift again at the next rebase; the
stage names are the stable key.)

**Adversarial audit, 2026-08-27** (5 auditors + triage over S0-S4): every
numeric claim checked against a primary source held, including the NVFP4 nibble
order and the FP8 decoder over all 256 codes. Three landmines sit where S5
steps, none in shipped output: (1) the 72.00 GiB slab commit exceeds the
**70.71 GiB driver budget** -- `ensure_fits` checks heap *size*, and the default
build compiles the dry-run out; (2) the `1 + w` RMSNorm fold is per-family, not
global -- GatedResidual `hc_norm` folds at load, but `qwen4_ple_gate` applies
`(1.0 + w)` in-shader so its three norms upload RAW; (3) `norm_topk_prob` is
absent from config.json and the HF default is `true` -- defaulting it false
attenuates every MoE output ~2.5x, finite and coherent-looking. Plus five
tests-that-cannot-fail, worst first: the NVFP4 repack fixture is periodic with
period 8 so index mutations pass; no test pins any tensor family to a
shape/dtype.

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

## On-chip reuse is the reason to prefer Config C

Measured, not quoted (`device_cache_hierarchy`, streaming read+write):

```
   12 MiB   285.1 GB/s  111% of DRAM peak
   16 MiB   300.8 GB/s  117%   <- last size served on-chip
   24 MiB   164.0 GB/s   64%   <- cliff
 1024 MiB   207.9 GB/s   81%   <- DRAM asymptote
```

The kernel reads and writes, so a 16 MiB working set puts ~32 MiB through the
cache — consistent with a 32 MB MALL, and the first time that number has been
measured on this box rather than taken from a spec sheet. The 81% asymptote
independently reproduces the GEMV efficiency measured on the 122B.

This model is unusually well-shaped for that tier, because `hidden = 2560` and
`moe_intermediate_size = 640` make everything small:

| per layer | BF16 | Q8_0 | Q4_K |
| --- | ---: | ---: | ---: |
| hyper-connections | 25.2 MiB | 13.4 | **7.1** |
| linear_attn | 110.5 MiB | 58.7 | **31.1** |
| 10 active experts (NVFP4, fixed) | — | — | **26.4** |

At Q4_K every piece is at or under the tier; at BF16 none of them are. That is a
second, independent argument for Config C beyond the bytes-per-token one — a
dense 27B layer is 235 MiB and has no such regime at any precision.

**Batching is the multiplier, and it applies to the dense 87%.** Weights read
once and used B times:

| batch | weight bytes/token | roofline |
| ---: | ---: | ---: |
| 1 | 9.95 GB | 25.7 tok/s |
| 4 | ~3.4 GB | ~76 tok/s |
| 8 | ~2.2 GB | ~117 tok/s |

The expert half does NOT amortize the same way: at top-10-of-512, a batch of 8
touches up to 80 distinct experts, so each serves ~1 token and its bytes are
re-read per token. The reuse comes almost entirely from the non-expert weights,
which is the same 87% that dominates B=1. This is why MTP earns its 1.41 GiB
twice over — batched verify turns one weight sweep into N tokens — and why it
must wait for a batched MoE forward rather than shipping against a sequential
one.

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
| S1 ✅ | strided router (512), NVFP4 dtype, FP8 E4M3 dequant | done |
| S2 ✅ | safetensors source, slab uploader, mmap'd N-table host embedding | done |
| S3 ✅ | NVFP4 fused-expert GEMV + CPU oracle | done |
| S4 ✅ | fused hyper-connection + PLE kernels vs host oracles | done |
| S5 ✅ | PLE + n-gram → **first token** (indexer stubbed, ≤2048 ctx) | done |
| perf ✅ | decode 899 → 84.9 ms/token: resident linattn/PLE/full-attn, staged loop (50 fences), BF16 tiers | done |
| prefill ✅ | chunked `forward_prompt` 11.8 → 31 tok/s, prefill=decode BIT-EXACT at full scale (0.000e0): seq-mode linattn/PLE, batched flash, chunk perm maps, per-token MoE + one ids fence per layer-chunk (87% of wall — S7's grouped experts are the 100 tok/s lever). Coopmat BF16 GEMM lane (61 tok/s) opt-in `ARLE_QWEN4_PREFILL_GEMM=1`: any sub-f32 staging saturates to an argmax flip at 48 layers (expert-flip boundaries) | done |
| Q4 ✅ | W4A16 Q4_K dense default: 79.0 → 55.5 ms/token (18 tok/s); teacher-forced step-agree 84.4% vs near-lossless Q8's 90.6% (2/32 gap at razor margins); the W4A8 q8_1-activation detour was built, measured, and deleted — A16 is the contract | done |
| S6 | QSA indexer → contexts >2051 | 40 h |
| S7 | requantize experts (Q8_0 returns MTP+vision); batched MoE prefill + verify (cols k≤8 / WMMA k≥16, measured) | 40 h |
| S8 ✅ | MTP speculative decode: greedy-LOSSLESS at full scale (15/15 configs, 3 prompt classes x 5 depths x 40 tok), acceptance 57.9-71.9%/step at k=2 (vendor band 50-70%), best k=**1** at +10-15% (46.0/46.3/49.6 vs 50.6/51.1/56.8 ms/tok). Rollback = 2 device-to-device copies, 0.19-1.90 ms/cycle. NOT a default flip — chat regresses at k>1. Measured next wall: the verify amortizes the DENSE tier 1.31x and nothing else (the 512-expert union grows ~linearly in k; the per-position hc/router dispatch floor does not batch) | done |

Hours are CC-execution, not calendar.

## Upstream (surveyed 2026-08-28)

llama.cpp merged qwen4exp on 2026-08-27 (PR #27742) — one day into this
plan's S7. Coverage: everything except MTP (WIP; the closed #27739 carries
the only public MTP-draft code). Its Vulkan path ships with qwen4exp-specific
breakage (ggml_vk_topk assert, Windows crashes #27431/#27560), so ARLE is
plausibly ahead on this exact platform — a same-box UD-Q4_K_XL llama-bench
baseline is being measured to replace that "plausibly" with a number.
Lifted intel: MTP n-max=2 / dense-attention drafting / one-batched-verify;
KV stays f16 under rotated QSA (S6 landmine); PLE graph-input reuse was
worth +22%@b16 upstream (ARLE's staged loop already amortizes this).

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
