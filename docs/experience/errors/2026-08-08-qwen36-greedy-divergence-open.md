# Greedy divergence from two external runtimes on Qwen3.6-27B-FP8 — root cause OPEN

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

**OPEN.** Localized to the model forward; every component that could be tested is
excluded below, and the route used cannot go further.

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

## Next step (not started)

**A component parity harness with a shared exact input.** Feed HF's layer-0
`post_attention_layernorm` output into our `dense_mlp` and compare outputs; same
for the GDN block with HF's `input_layernorm` output. Input error is zero by
construction, so the output difference is that component's own and there is no
accumulation to confound it. The pattern exists at
`crates/infer-cuda/examples/marlin_w8a16_parity.rs` and `dsv4_parity.rs`.

## Rule

- **Convert to absolute before comparing tensors of different norms.** Per-tensor
  `relL2` made a 0.36-norm buffer look worse than a 10.0-norm one, and would also
  have shown a fake 10× GDN amplification.
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
