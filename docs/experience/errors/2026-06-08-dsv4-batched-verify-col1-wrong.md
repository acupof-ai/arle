# DSv4 batched MTP verify: +63% speed but col1 (draft position) computed wrong

## Context

MTP at B=1 is net-slower because `forward_tokens_verify` loops two single-token
forwards (reads the 149GB weights 2×). Implemented an opt-in batched path
(`ARLE_DSV4_MTP_BATCHED_VERIFY`, default off): ONE
`forward_tokens_stream_impl(&[pending, draft], start_pos)` + sample col0=base_next /
col1=bonus, with `capture_spec_rollback(start_pos+1)` before the batch and a reject
path of `truncate_slot(start_pos)` + restore + re-forward pending.

## What the A/B showed (8×H20, two prompts, vs spec-off ref)

- **Speed: the amortization works.** Batched verify hit **62.9 tok/s on needle
  (+63% vs spec-off 38.5, +87% vs per-token 33.7)** and 50.9 on capital (+28%). A
  2-token forward ≈ a 1-token forward (memory-bound) — confirmed.
- **Correctness: FAIL.** Batched output diverges from spec-off EARLY on the
  non-degenerate capital prompt: off/pertok both `[11111,14,778,…]`, batched
  `[11111,14,260,…]` (position 2). Per-token spec is byte-identical to spec-off, so
  this is a real bug, not MoE non-determinism.

## Root cause (diagnosed, not yet fixed)

col0 (base_next, the pending position) is **correct** (14 matches); col1 (bonus, the
draft position) is **wrong** (260 vs 778). So the **second token of the 2-token
forward is computed incorrectly** — the reject rollback is not even involved (position
2 is an accept). The prefill driver's multi-token path at `start_pos > 0` (a 2-token
"chunk" attending to KV history + intra-chunk) does not reproduce what two sequential
single-token forwards produce for DSv4's compressed attention — likely the DSA
compressor / sliding-window / head-HC sees the draft (token 2) without token 1's
sequential update. This is the chunked-prefill-at-start_pos>0 path for compressed
attention.

## Root cause localized further (2026-06-08)

`head_hidden_from_stream(stream, token_idx)` extracts the row correctly
(`copy_row_to_hidden(stream, token_idx)`), so the output-extraction is NOT the bug —
**the 2-token forward computes the draft's stream row (col1) wrong**. The smoking gun:
`forward_tokens_stream_impl(seq_len=2, start_pos>0)` is a combination the normal flow
**never exercises** — prefill is always a single chunk at start_pos=0; decode is
seq_len=1 at start_pos>0. The batched verify is the FIRST caller of seq_len>1 AT
start_pos>0 (a "decode-position chunk"), so it hits a latent bug in the
sliding-window/compressed attention's handling of a multi-token chunk attending to KV
history at start_pos>0 (token 0 / col0 is correct; token 1 / col1 — which must attend
to both the history AND token 0 — is wrong). The fix is in the chunked-prefill SW/CSA
attention path, not the verify wiring. Re-validate with a planted-answer needle.

## Suspects ruled out by source read (2026-06-08) — narrows the pod probe

- **head_hidden_from_stream**: extracts `token_idx` correctly (`copy_row_to_hidden(stream, token_idx)`) — output extraction is fine.
- **FlashMLA decode**: `try_flashmla_decode_attention` returns `Ok(false)` for `seq_len != 1` (attention.rs ~3291), so the 2-tok chunk correctly falls back to the legacy SWA kernel — NOT a FlashMLA s_q=1 mishandle.
- **SWA kernel causal** (`dsv4_swa_attention_kernel`, dsv4_attention.cu:539): for token 1 (abs_pos=start_pos+1), `sw_start`/`key_count` (566-568) give key_pos 0..start_pos+1, and `dsv4_swa_key_value` routes key_pos≥base_start_pos → `k_new[key_pos-base_start_pos]` — so token1 DOES attend to token0 + history. Chunk math is correct.

So col1 is wrong somewhere ELSE in the 2-token forward's token-1 path (HC mix `gen_mhc_params`/`hc_pre`/`hc_post` over seq_len=2, MoE token-1 routing, or a window_cache write-then-read ordering within the chunk). NOT crackable by reading — needs a pod probe that dumps token-1's per-stage hidden vs the per-token reference (the working `forward_tokens_stream_impl([draft], start_pos+1)`) to find the first diverging stage.

## Rule

- A batched K+1 verify is the right MTP lever (speed proven +63%), but it requires the
  multi-token forward at `start_pos>0` to be **bit-equivalent to sequential single-token
  forwards** for compressed attention — verify that first (dump col1 hidden vs a
  sequential draft forward) before wiring the verify. Gate on **greedy-identity on a
  NON-degenerate prompt** (capital), not the degenerate needle loop (confounded by MoE
  non-determinism per `feedback_correct_inference_not_baseline_identity`).
- Reverted the code (default-off but incorrect = a half-state); re-implement after the
  chunked-prefill col-N correctness is established. Design is in
  `reference_dsv4_decode_6ms_path_state`.

## PINPOINTED to the attention (2026-06-08, ARLE_DSV4_TAIL_DUMP probe)

Per-layer-0 tail-row (token-1) dump, batched (sp=5 seq=2) vs per-token ref (sp=6 seq=1):
- `init_stream`: l2=2.86334, first4=[0.0270,…] — **BIT-IDENTICAL** (embed fine)
- `attn_in_L0` (post HC-pre): l2=5.72668, first4=[0.1079,…] — **BIT-IDENTICAL** (HC fine)
- `attn_out_L0` (post attention): batched l2=144.86 first4=[-1.10, 2.125, -2.125, 3.469];
  ref l2=145.18 first4=[-1.20, 1.953, -2.0, 3.375] — **DIVERGE**, concentrated in the
  first/RoPE elements.

So the col1 bug is the **SWA attention for the chunk's 2nd token** (not embed/HC/MoE/
output-extraction). The kernel's output inverse-RoPE uses `abs_pos=base_start_pos+token`
correctly, and the inputs (token_a via k_new[0] vs ring[5], history ring[0-4], query,
sink) all appear to match on read — so it is a subtle multi-query SWA effect. It's a
real spec-decode-contract bug (the verify must reproduce non-spec greedy, so a
"legitimate numerical" path difference still breaks losslessness). Next dump: k_new[0]
(batched token_a key) vs ring[5] (per-token token_a key) to confirm the prepare/store
path. The ARLE_DSV4_TAIL_DUMP instrumentation (dsv4.rs dump_tail_row) is committed for
this.

## Inputs proven bit-identical → bug is INSIDE the SWA kernel (2026-06-08)

k_prepared row dump (ARLE_DSV4_KNEW_DUMP), layer 0, batched [token_a,wrong_b]@5 vs
the per-token references:
- token_a key: batched row0 (sp5 seq2) == per-token [token_a]@5 row0 (sp5 seq1) —
  l2=15.76101, first4=[-0.165,0.402,0.142,0.041] — **BIT-IDENTICAL**.
- wrong_b key: batched row1 (sp5 seq2) == per-token [wrong_b]@6 row0 (sp6 seq1) —
  l2=13.90704, first4=[0.836,-0.295,-0.270,0.080] — **BIT-IDENTICAL**.

So q/k prepare is NOT the bug. Combined with attn_in_L0 bit-identical (earlier) and the
history ring coming from the same deterministic prefill, EVERY input to token-1's
attention is bit-identical, yet attn_out_L0(token-1) diverges deterministically
(144.86 vs 145.18). Every host-visible index/path in `dsv4_swa_attention_kernel`
(q_base=token*local_width+head*head_dim; k_new[(key_pos-base_start_pos)*head_dim+col];
sw_start/key_count; sink_offset+head; output inverse-RoPE at abs_pos=base_start_pos+token;
ring write head==0) is correct on read. So the bug is a seq=2-specific compute issue
INSIDE the kernel for the 2nd query — not host-visible. Next: kernel-side printf of
token-1's per-key logits / out_vec (pre-inverse-RoPE), or compare the batched seq=2
kernel launch against two seq=1 launches numerically with a minimal in-kernel probe.
This is the deepest host-side localization possible; the remaining search is one CUDA
kernel's multi-query path.

## RESOLUTION (2026-06-08): col1 is legitimate FP8 KERNEL-PATH numerics, not a logic bug

Elimination chain (all dumps committed): inputs bit-identical (q+k+attn_in+history);
pre-all-reduce attn_out ALREADY diverges (not NCCL AR); FlashMLA-decode OFF still
diverges. The remaining difference is the KERNEL PATH between the batched verify (M=2)
and non-spec (M=1):
- attention: batched=SWA-prefill kernel (seq>1); non-spec/per-token=FlashMLA-decode
  (seq==1).
- wo projection: batched=`prefill_proj_deepgemm` (M=2 DeepGEMM); non-spec=
  `decode_proj_deepgemm` (M=1 DeepGEMM) — DIFFERENT DeepGEMM kernels.

Different FP8 kernels round differently; over 43 layers this amplifies to the col1
token flip (deterministic, 0.2% per-stage). This is NOT a logic bug — it's the same
legitimate FP8-tensor-core numerics class as the projection-DeepGEMM work, and the
byte-identity-vs-non-spec gate is CONFOUNDED by it (per
`feedback_correct_inference_not_baseline_identity` / `reference_dsv4_moe_nondeterminism`).
I nearly "fixed" a non-bug — the same trap the memory warns about.

## Corrected rule

- The MTP batched-verify gate is **determined-answer needle retrieval** (correct
  inference), NOT byte-identity vs the per-token/non-spec path. A batched K+1 verify
  inherently runs M>1 kernels (SWA-prefill, prefill-DeepGEMM) that differ in FP8
  rounding from the M=1 non-spec kernels (FlashMLA-decode, decode-DeepGEMM); bit-
  losslessness is unachievable and not required. Validate with a planted-answer
  long-context needle (CSA-active) + a same-config-twice non-determinism floor.
- Next: re-implement the executor batched-reject (capture-before + truncate + restore
  + re-forward pending) and run the full MTP loop through a determined-answer needle;
  if retrieval holds within the non-det floor, license MTP (+63%) and flip default-on.
