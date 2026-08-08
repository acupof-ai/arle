# The final norm applied `w` instead of `(1+w)` on Qwen3.6-27B

**Date:** 2026-08-08 · **Pod:** 8×H20 · **Model:** ThinkingCap-Qwen3.6-27B-FP8
(dense, 64 layers = 48 GDN + 16 full-attention, `full_attention_interval: 4`,
`head_dim: 256`, `vocab_size: 248320`, FP8 e4m3 block `[128,128]`, dynamic
activation scale)

## Context

GSM8K in the capability lane returned 27.4% `extract_fail` (mean over 5 seeds,
range 25.8–29.2%). The failures are not an extractor bug: the model emits a
degenerate repetition run and never reports its answer. ~79% of invalid items are
that degeneracy — **22% of all 500 GSM8K items**. Base capability point for the
record: MMLU 0.8693 ± 0.0170 (invalid 5.2%), GSM8K 0.8981 (invalid 27.4%),
`--concurrency 8`, n=500/task, seeds 0–4, whole grid 2.34 h.

Chasing one failing item produced a deterministic repro and a hard result: **our
runtime's greedy argmax disagrees with two independent external runtimes on an
identical 130-token prefix.**

## The repro

Prompt: GSM8K test split, `random.Random("gsm8k-0").shuffle`, pool index 0
(gold 410). Instruction sha256 prefix `a66d6b171683a5841b716317`, 384 chars.
Chat-templated to **108 prompt token ids**; the first **22 generated ids agree**
across all three runtimes. The comparison state is those **130 token ids**, fed by
id so no re-tokenization can drift (`prompt_token_ids` round-trip verified
`IDENTICAL=True` on every run).

At generated position 22 (absolute position 130):

| runtime | precision | top-1 | p(top-1) | p(328) | p(5289) | entropy |
|---|---|---|---|---|---|---|
| sglang 0.5.13 | its own FP8 | **328** | 0.768 | 0.768 | 0.081 | ~1 nat |
| HF transformers 5.6.0 | dequantized bf16 | **328** | 0.7639 | 0.7639 | 0.1034 | 0.7085 nats |
| **ARLE** | FP8 | **5289** | **0.350** | — | 0.350 | **7.147 nats** |

**The two external runtimes bracket the question**: one in FP8, one in bf16,
agreeing to **0.004 in p1** and both on `328`. So FP8-versus-bf16 is worth ~0.4%
at the output on a correct implementation, and ARLE's 0.35 is a real divergence,
not quantization noise. ARLE's own top-8 has `5289` at rank 1 (logit 12.75, margin
1.6875 over `328` at 11.0); in sglang's distribution `5289` is **rank 3, 2.25 nats
down**. Both runtimes are internally confident and disagree about the winner —
not a near-tie amplified by reduction order.

Downstream, ARLE's trajectory degenerates: 476 tokens later it emits token
**151353** (`ª½`, bytes `aa bd`, bare UTF-8 continuation bytes = the tail of a
4-byte emoji) **48× consecutively**, `finish_reason=stop`, no answer. Zero
byte-fallback tokens and zero 4-byte lead bytes in the whole 546-token
generation. The loop is a symptom of the divergence, not the defect.

## Root cause

**The trunk's final RMSNorm before `lm_head` applied `w` where the model's
convention is `(1 + w)`.** Qwen3.5/3.6 uses the offset convention, `norm.cu:666`
carries the kernel family for it (`// RMSNorm with (1+weight) offset — Qwen3.5 /
Gemma style`), and the 64 in-layer trunk norms already called it. So this was a
call-site selection error, not a missing implementation — which is exactly why
in-layer norms were bit-exact (`attn_in` relL2 1.7e-09) while the final one was
2.1× off.

| norm convention | ‖norm_out‖ | argmax | p1 | logit[328] | logit[5289] | rel vs HF logits |
|---|---|---|---|---|---|---|
| `x·rsqrt(…)·w` | 56.2240 | **5289** | 0.3282 | 11.0752 | 12.6197 | 9.987e-01 |
| **`x·rsqrt(…)·(1+w)`** | **118.1459** | **328** | 0.7659 | 28.0149 | 25.9583 | **2.168e-03** |
| HF actual | 118.1514 | 328 | 0.7639 | 28.0 | 26.0 | — |

The scale is the giveaway: 118.1514 / 56.2240 = **2.101**, and the norm weight's
RMS is 0.9715, so `(1+w)` has RMS ≈ 1.99. `norm_in` is bit-identical to HF's L63
residual (`‖d‖ = 0.000e+00`), so the divergence is created inside the norm.

Found in numpy from tensors already on disk, before any source was read —
reproducing the wrong answer by omitting the `+1` is what named the defect.

## Fix

`694245eec` — 14 call sites swapped to `rms_norm_offset` / `rms_norm_offset_vec`
(`qwen35.rs:9219`, `:9246`; byte-identical signatures, and the latter's doc
comment already read "the final norm before lm_head").

**Two instances, not one.** The trunk's final norm (`qwen35.rs:4359` plus six
batched/OPD/spec sites on `&self.norm`), and separately **every norm in
`qwen35/dspark.rs`** — `input_layernorm`, `post_attention_layernorm`,
`head.norm`, `head.hidden_norm`, the same weights the trunk applies `(1+w)` to.
The MTP head's final norm (`:4868`) was already correct, so the plain-`w` variant
had no remaining legitimate caller on this model. The DSpark instance bears on
spec-decode rather than on this repro and a trunk-only investigation would never
have surfaced it.

**Verified, gate 1** (`694245eec`, same 130 ids, `IDENTICAL=True`, single prefill,
greedy): top-1 **328**, top-3 **328 > 348 > 5289** — the exact ordering both
external runtimes give, with `5289` moved from our rank 1 to rank 3. Logits
27.875 / 26.25 against HF's 28.0 / 26.0, bf16 granularity at that magnitude.
`nan=0 pos_inf=0 neg_inf=0`.

**Gates still pending:** the GSM8K item end to end (does it answer instead of the
48× `ª½` run), needle ×3 same-config against the baseline envelope, then the
capability re-measure on the identical grid. This changes the logits of **every**
request, so the needle gate and the capability delta are what license it — not
this one position.

## Exclusions, each with the evidence that closed it

| Excluded | Evidence |
|---|---|
| Streaming detokenizer | Separate real bug, fixed (`ee6339fd7`, [entry](2026-08-08-streaming-detokenizer-splits-multibyte-codepoints.md)); the U+FFFD runs here are literal replacement chars and the raw body decodes as strict UTF-8. Verified fixed: streamed text 0 U+FFFD and byte-identical to non-streaming. |
| NaN / Inf in logits | `finite=248320 nan=0 pos_inf=0 neg_inf=0` on 4 runs, prefill and decode. |
| Decode path (paged KV, KV reuse, decode attention, per-step state) | A **single prefill forward** over the 130 ids reproduces it: top-1 `5289`, p1 0.363, entropy 7.147 — against decode's 0.350 / 7.308. Two paths agree to **0.06 logits**. |
| fq-chunked versus recurrent GDN | `ARLE_FQ_PARITY=1`, 48/48 GDN layers: `state_rel` median 1.3e-2, `o_rel` median 7.0e-3. Sampled token, logits, p1 and entropy **byte-identical** with parity on. |
| MoE router | **No MoE in this checkpoint** — no `num_experts` / `num_experts_per_tok`; `intermediate_size: 17408` is a dense MLP. |
| RoPE | `rope_parameters.rope_theta = 1e7` parsed as a required field (`qwen35-spec/src/lib.rs:905`); mrope sections carry equal positions for text, so interleaving is a no-op. |
| Attention output gate | sglang applies `torch.sigmoid(gate)` then multiplies **before** `o_proj` (`qwen3_5.py:862-867`); ours is identical, and the per-head interleaved addressing matches (`q_head*2*head_dim + head_dim + dim` against `view(...,num_heads,-1)` + `chunk(2,-1)`). `output_gate_type: "swish"` is read by **neither** runtime (0 hits in sglang). |
| FA3 hd256 prefill shim, and with it the full-attention block | Same-binary A/B `--qwen35-fa3 false`. Path swap **proven** by profile timings (steady state 0.031→0.022 ms, first launch 0.851→0.133 ms, 32 calls each) — the serve takes the *paged* gate at `qwen35.rs:6731`, not `:6310`. Top-1 stays `5289`; logits move ≤0.125, p1 0.363→0.352. |
| FP8 GEMM kernels | Prefill (M=130, DeepGEMM dense) and decode (M=1, scalar GEMV lane) are **different kernels** and agree to 0.06 logits while both sit 1.7 logits from the reference. |
| FP8 load-time dequant / block-scale metadata | Independent numpy dequant against our `n_block*k_blocks + k_block` indexing: **max_rel = 0.000e+00, bit-identical**, across 8 tensors and every projection shape ([12288,5120], [1024,5120], [5120,6144], [17408,5120], [5120,17408]). Scale grids are all non-square, so a transposed index cannot be silent. `weight_scale_inv` → `Multiply` confirmed (`quant_format.rs:118-128`); dequant range [−0.2478, 0.4067]. |
| Embedding | relL2 **9.28e-09**, norm ratio **1.00000000**, cosine 1.0, max element diff 5e-09. The input is exact; layer 0's error is generated inside layer 0. |
| Dense MLP, all three ops | Component parity with an exactly injected `mlp_in`, against an **FP8-emulated** reference (our own scale rule applied to `bf16(x)`, then a bf16-dequantized weight in f32): `gate_up` gate half relL2 **2.053e-03**, up half **2.069e-03**, both **1.1× floor**, cosine 0.9999979. Against HF bf16 the same halves read 7.1× / 7.4× floor — and **HF-bf16 against the FP8-emulated reference is itself 7.1× / 7.3×**, so the whole gap is what FP8 costs, not ours. Control: with the activation left un-quantized, HF-bf16 and the reference agree at the floor (gate 1.654e-03), so the gap comes from activation quantization specifically rather than from the weight-dequant path. |
| A quantization-recipe difference against sglang | sglang 0.5.13 takes `Fp8Config` → `Fp8LinearMethod.apply` → `w8a8_block_fp8_linear` (`fp8.py:165-183`, `:732`, `:791-799`), dispatching to **DeepGEMM** on H20 with no bf16 fallback for these shapes. It quantizes activations **per-token-group, group 128**, scale `amax / 448` with a clamped amax floor (`fp8_utils.py`, `fp8_kernel.py:498`, `:127-132`). **The same granularity and the same rule as ours** (`dsv4_deepgemm_ops.cu:88-117`). Not checked: the `DEEPGEMM_SCALE_UE8M0` default, which would make sglang's scales coarser, not finer. |

## What the residual-stream comparison does say

Per-layer at position 129, ours against HF (dequantized bf16), full 5120-wide rows:

- **Magnitude, not direction.** Cosine ≥ 0.9988 at every one of 64 layers. L0–L1
  error is pure direction (norm ratio 1.00002 / 1.00004, relL2 1.2e-03 / 1.9e-03);
  from L2 magnitude dominates (ratio 1.01186, our residual ~1.2% larger). The
  ratio **flips sign around L23** (1.012 → 0.996) — unexplained.
- **Depth profile:** relL2 8.7e-03 (L0–7) → 1.7e-02 (L16–23) →
  **4.7e-02 (L32–39)** → 2.9e-02 (L56–63). Injected early, compounds, then
  saturates. **No step at any layer**, so no layer is named.
- **GDN and full-attention are indistinguishable:** relL2 median **gdn 2.852e-02
  (n=48)** against **FULL 2.908e-02 (n=16)**. Both block types carry it equally,
  so "GDN is all that is left" reasoning is **wrong**.
- Per-block amplification in absolute terms: L1 GDN 1.34×, L2 GDN 1.07×, L3
  full-attention 0.31×. No block amplifies anomalously; full-attention attenuates.

## Retracted claims

1. **"82 of 500 GSM8K items contain a `####` the extractor failed to parse"** —
   measured by re-running on `/v1/completions` while the harness uses
   `client.chat`. On the correct chat path there are **zero** unparseable `####`
   items. The dominant bucket is 90/140 **empty content** (79/90 finish `stop`,
   median 304 completion tokens, so not budget exhaustion): the same degeneracy
   occurring inside the thinking block. **There is no extractor bug.**
2. **"The dense MLP is the entry point"** — from ranking taps by `relL2`, which
   normalizes by each tensor's own norm; `mlp_out` has ‖·‖≈0.36 against
   `attn_out`'s ‖·‖≈10.0. In **absolute** error the attention block contributes
   1.8× more than the MLP at L0 (0.01152 against 0.00640), 1.7× at L1, 2.3× at L3.
   **No component is named.** The same small-denominator trap appeared three times
   in one investigation.
3. **"The dense MLP is 6.4×–11.6× its floor, so it is the defect"** — the floor
   was `‖a‖ × 2⁻⁹`, a **bf16** floor, applied to a GEMM whose activations are
   **e4m3**. Three mantissa bits give a per-element RTN relative error around
   3.6% RMS, and the measured 1.4e-2 is *below* that crude prediction. Against an
   FP8-emulated reference every `gate_up` half sits at 1.1× floor. **The
   reference class was wrong, not the metric** — the same structural failure as
   the accumulated-state route.
4. **"If sglang paid 1.4e-2 per GEMM it could not land within 0.5% of HF at the
   output, so its recipe must differ"** — the premise assumed per-GEMM error
   compounds across layers. The depth profile refutes it: relL2 goes 8.7e-03
   (L0–7) → 4.7e-02 (L32–39) → **2.9e-02 (L56–63)**, saturating and then
   decreasing. The error acts on each block's contribution, not on the residual
   stream itself, so 64 layers do not give 64 × 1.4e-2. sglang paying the same
   per-GEMM cost **is** compatible with 0.004 agreement at the output.

## Limits of this route

- **The bf16 floor is the model's arithmetic, not the instrument.**
  `HiddenStates.data` is bf16 on device, so `resid` genuinely *is*
  `bf16(resid_mid + mlp_out)`. The taps' additive-identity residual is predicted by
  bf16 rounding with **ratio exactly 1.00** at L0–L3, and HF's rows satisfy
  `max|v − bf16(v)| = 0`. An f32 D2H would add digits, not information.
- Consequence: at L0–L1 the divergence has not grown above bf16 granularity, so
  there is nothing to attribute; from L2 the signal clears the floor but every
  input is already polluted, so attribution is confounded. **Comparing accumulated
  states cannot localize this** — a structural limit, not a metric choice.
- The triangle check `‖Δresid‖ ≤ ‖Δresid_mid‖ + ‖Δmlp_out‖` passes at L0/L1/L3 and
  **fails at L2** (0.1285 against 0.1066 including the floor). The cause is the
  reconstructed `resid_mid` reference (HF exposes no post-attention-add hook; its
  own additive identity at L2 carries 4.5e-02 of slack), so **L2's reading is
  invalid**. Not an indexing bug: `hf_resid.json` and `hf_taps.json` agree
  bit-for-bit at L0–L3.

## Consequences already actionable

- **The GSM8K invalid rate measures this defect, not model capability.** The
  full-1319 GSM8K point is on hold; MMLU stays the capability gate. Note MMLU's
  **3.9 pp seed-to-seed spread** (sd 0.0170): a per-round adapter delta must be
  read as a per-question paired comparison on the same seeds
  (`arle_capability_eval.py:420`, `mmlu_perquestion.json`), not a difference of
  means.
- **The `fulltrain11` production run continues.** `rejection-ce` trains only on
  pytest-verified passes, so a degenerate rollout fails its tests and is filtered.
  The defect costs pass yield and rollout budget, not training-data correctness.

## What the component parity harness settled, and what it left

The harness worked: `ARLE_PARITY_INJECT` / `_POINT` / `_LAYER` overwrite a tapped
buffer with a reference tensor (`probe.rs:549`, chunk-aware — prefill splits 130
into 128+2 and the injection must slice per chunk), so the component under test
gets a zero-error input and there is no accumulation to confound the output. With
it, **every component that has been tested sits at its floor against the correct
reference.** The dense MLP was the last candidate the residual-stream route had
pointed at, and it is exonerated.

**That result was the finding, not a dead end.** "Every component is at its floor"
closed the whole class "the defect is inside some kernel", and what remained was
the one segment with no tap on either side of it: between L63 `resid` and the
logits. The two tests that followed settled it:

1. **Is the position ill-conditioned?** No. Perturbing HF's layer-63 hidden at
   relL2 0.029 / 0.05 / 0.1 / 0.2 / 0.4 / 0.8, 12 draws each, then running HF's
   own final norm + `lm_head`: the argmax **never** moved and entropy stayed
   7.50–11.4. Generic error of any magnitude does not produce this flip. A
   specific systematic transform does.
2. **Final norm + `lm_head`.** The `(1+w)` result above.

**Weakened to not worth a run:** a position off-by-one in the logits row. It
would explain the entropy gap (7.147 nats against 0.7085 — a perturbed
distribution does not gain 6.4 nats, a different position's distribution does),
but prefill-130 and decode-at-130 agree to **0.06 logits**, far too tight for two
paths to be independently off by a row and coincide, and L0 `attn_out` at 0.01153
against a 0.01958 floor with an exact injected input rules out attention dropping
a key.

## Rule

- **Tap both ends of every segment, or the untapped one is where the bug lives.**
  Eleven exclusions and a component parity harness all landed inside the trunk
  because that is where the taps were. The final norm had a tap before it (L63
  `resid`) and nothing after it until the logits, so it was the only transform in
  the model no comparison could see — and it was the defect. Before the next
  round of probes, list the segments with no observable on one side.
- **A parity harness that puts everything at its floor has told you something.**
  It closes the class "some kernel is wrong", which is what makes the untapped
  segment the answer by elimination. Reading that result as failure cost a
  round trip.
- **The floor must match the arithmetic under test.** A bf16 floor on an FP8
  GEMM manufactured a 7× "defect" that was entirely FP8's own cost. The check
  that catches it is the **third comparison**: reference-A against reference-B.
  If the two references differ from each other by the same amount our output
  differs from either, nothing has been measured.
- **Convert to absolute before comparing tensors of different norms.** Per-tensor
  `relL2` made a 0.36-norm buffer look worse than a 10.0-norm one, and would also
  have shown a fake 10× GDN amplification.
- **Do not assume per-layer error compounds.** Block-local error acts on the
  block's contribution, not on the residual stream, and the observed profile
  saturates and then decreases. An argument of the form "N layers × per-layer
  error" is wrong by default and needs the depth profile to license it.
- **Grep the callers before changing a signature.** `dense_mlp` has six; two live
  in `qwen35/dspark.rs`, which is entirely behind the `cuda` feature and
  therefore invisible to the Mac `--no-default-features` typecheck. This is the
  fourth diagnostic in this investigation whose call sites were incomplete.
- **A reference is only a reference on the path under test.** Two separate
  endpoint confounds (raw against chat-templated prompt; completions against chat
  harness) each produced a confident wrong conclusion.
- **Prove a same-binary flag actually swapped the path** before reading its A/B —
  the FA3 gate exists at four sites and the serve takes a different one than the
  dispatch comment suggests; profile timings were the only proof available.
- **Check a diagnostic is wired for the model under test.**
  `ARLE_PROBE_LENS_LAYERS` and `ARLE_PROBE_STAGES` both had call sites only in
  `dsv4.rs` while their `meta` line printed as if armed — a silent empty result
  reads as "no divergence found". Fixed for both
  (`7b34f7fff`, `982175818`, `d708f216e`).
- Scripts named after stdlib modules (`bisect.py`) on the CWD shadow the stdlib
  and execute on import; this corrupted one measurement before it was caught.
