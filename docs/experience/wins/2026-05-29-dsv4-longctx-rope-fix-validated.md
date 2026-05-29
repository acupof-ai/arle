# DSv4 long-context correctness FIXED — RoPE conflation + missing FlashMLA output inverse-rope

## Context

DSv4-Flash (8×H20 TP=8) only produced correct output while the whole sequence
fit in `sliding_window=128`; beyond that it collapsed (repeat loops, no needle
retrieval, garbage). Root-caused and fixed across two stages; validated on the
default FlashMLA path. Supersedes the misdiagnosis in
[`../../projects/2026-05-29-dsv4-beat-sglang-30pct-campaign.md`] I3-v2 (the
shared KV pool was wrongly blamed; it is exonerated — both pool modes behaved
identically because the bug was in shared attention code). See
[`../errors/2026-05-29-dsv4-longctx-rope-conflation.md`].

## What was wrong (two independent defects, both in the compressed/long-range path)

1. **Q/SW-K RoPE conflated with the compressed-key theta.** `finish_attention_gpu`
   selected the RoPE base+YaRN for **Q and the sliding-window K** by layer mode:
   `compress_ratio>0 → compress_rope_theta(160000)+YaRN` instead of the main
   `rope_theta(10000)` no-YaRN. The CPU reference (`reference.rs`, the test
   ground truth) rotates Q (`:307`) and `kv_sw` (`:318`) **uniformly with
   rope_theta for all layers**; only compressed keys use `compress_rope_theta`
   (`:344`). The mis-rotated Q collapsed attention once long-range/compressed
   keys joined the softmax (seq>128).

2. **FlashMLA decode + prefill missing the attention-output inverse-rope.**
   `reference.rs:417` applies `apply_partial_rope(dst, .., sign=-1.0)` to the
   attention output (un-rotating the key-position rope carried in the value
   tail at the query position). The legacy `dsv4_hybrid_attention_cuda` does
   this internally; the **default** FlashMLA SM90 sparse decode/prefill shims
   do **not**. The MODEL1 kernel (head_dim=512) keeps the rope tail in V, so the
   FlashMLA output carried un-cancelled key-position rope. Changing the input
   theta (fix #1) merely shifted which positions greedy-argmax survived, so it
   *regressed short context* on the FlashMLA path until #2 landed.

Plus a parity nit: the CSA/HCA causal block count was `(abs_pos+1)/ratio` vs
the reference `floor(abs_pos/ratio)` — fixed in the legacy, decode, and
(finally) prefill index builders.

## What worked (the fix)

`infer/src/model/deepseek/weights.rs`:
- Q/SW-K (and legacy output) rope: always `(rope_theta, 0)` — no per-mode gating.
- Compressed-K build: `compress_rope_theta`, `original_seq_len=0` (no YaRN).
- New: call `arle_dsv4_output_inverse_rope_cuda` after the FlashMLA **decode**
  fwd (token_count=1) and after the FlashMLA **prefill** fwd + TP-out-slice
  (per-token), on `out_ptr`=local_attn (the buffer feeding wo_a), for both
  TP=1 and TP>1.

`crates/cuda-kernels/csrc/misc/dsv4_attention.cu`:
- New kernel `arle_dsv4_output_inverse_rope_cuda` — one thread per RoPE pair,
  last `qk_rope_head_dim` cols, `sign=-1.0`, `rope_theta`, no-YaRN,
  `abs_pos=start_pos+token`. Mirrors the legacy output rope (`:966-981`).
- CSA causal count `abs_pos/ratio`.

`crates/cuda-kernels/csrc/misc/arle_flashmla_csa_prep.cu:174`: HCA causal count
`abs_pos/ratio`. `crates/cuda-kernels/src/ffi/misc.rs`: FFI binding.

Commits: `d61d26f4` (input rope), `8105d5c6` (HCA causal), `003c8370`
(FlashMLA output inverse-rope + prefill HCA).

## Validation (8×H20 TP=8, pod-built `target-pod/release/infer`)

Needle-in-haystack (code at prompt start, question at end, greedy, FP8 KV):

| path | 40 | 128 | 130 | 247 | 409 | 580 | 1147 | 2047 |
|---|----|-----|-----|-----|-----|-----|------|------|
| **default FlashMLA (decode+prefill)** | HIT | miss¹ | HIT | HIT | HIT | HIT | HIT | **HIT** |
| legacy (FlashMLA off) | HIT | — | — | HIT | — | — | — | HIT |

Default FlashMLA path: **11/12 HIT across two needle codes (ZORBLAX7, QUBIT42),
prompt_tok 100–2047** — was *all MISS / garbage* before. ¹The lone miss is at
*exactly* 128 (the SW boundary); 130 retrieves. Before the fix every length >128
collapsed; the legacy cross-check (input-rope fix only) already retrieved
40/247/922/2272.

**Decode-tail degeneration is universal greedy behavior, NOT a FlashMLA bug.**
A/B on the *same* open-ended `ignore_eos` 200-token story prompt: the
reference-correct **legacy** path generates ~120 tokens of coherent prose
("…a colossal spike of granite named Aethel's Peak, was his home, his charge,
and his prison…") then degenerates into "a, a, a"; the FlashMLA path does the
same a bit earlier (~70 tokens) — the difference is FP8 KV precision (FlashMLA
reads the FP8 pool, legacy reads bf16), an expected quant tradeoff, not the bug.

## Rule

- RoPE base/scaling is per-TENSOR-ROLE (Q & SW-K & output = main theta;
  compressed-K = compressed theta), NOT per-layer-mode.
- A new accelerated attention path must replicate EVERY step of the reference,
  including the **output inverse-rope** — diffing the fast path against a
  reference-correct slow path (legacy here) on an unambiguous probe (needle
  retrieval at seq≫sliding_window) is the gate, not "looks coherent" at short
  shapes.
- Distinguish a real decode bug from greedy/`ignore_eos` degeneration by A/B
  against the known-correct path on the SAME prompt before blaming the diff.

## Next (campaign)

Correctness gate now PASSED on the default path → the beat-SGLang throughput
work (per-step decode cost, batched attention, allreduce overlap) can resume on
a correct foundation. FP8-KV early-degeneration vs bf16 is a separate quality
lever to quantify. Throughput bench entry to follow.
