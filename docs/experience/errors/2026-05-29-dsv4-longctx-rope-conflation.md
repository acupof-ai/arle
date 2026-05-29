# DSv4 long-context collapse — Q/SW-K RoPE conflated with the compressed-key theta

## Context

DSv4-Flash (8×H20 TP=8) produced coherent output ONLY while the whole
sequence fit inside the sliding window (`sliding_window=128`). The moment
`seq > 128` it collapsed:

- short prompt + greedy 200 tokens → coherent story up to absolute position
  ~128, then a tight repeat loop (`"The second bullet hit the dead man."`).
- long prefill (>128) → garbage (`xxxxx`) / recent-token echo; a needle
  (`ZORBLAX7`) placed anywhere — even in the last 40 tokens of a long prompt —
  was NOT retrievable.
- IDENTICAL with `ARLE_DSV4_SHARED_KV_POOL=0` and `=1`, so the KV pool was a
  red herring (the shared pool matches the per-state pool byte-for-byte; it is
  the industry-standard PagedAttention design and short-context parity already
  PASSed). See [[project_dsv4_compressed_attention_longctx_bug]] and
  [[feedback_unvalidated_path_not_reference]].

The "foundation win 3/3 coherent output" (2026-05-29) only ever validated
<128-token prompts, where the SW window covers the whole sequence and the
compressed / long-range attention path is never exercised — so the bug shipped
unnoticed.

## Root Cause

In `finish_attention_gpu` (`infer/src/model/deepseek/weights.rs`), the RoPE
base + YaRN for **Q and the sliding-window K** were selected by layer mode:

```rust
let (rope_base, original_seq_len) = if compress_ratio > 0 {
    (self.config.compress_rope_theta, rope_params.original_max_position_embeddings) // WRONG
} else {
    (self.config.rope_theta, 0)
};
```

This `(rope_base, original_seq_len)` feeds `dsv4_prepare_qk_*` (Q and SW-K
input rope) AND the legacy `dsv4_hybrid_attention_cuda` output inverse-rope.
So in every compressed layer (`compress_ratio` 4 or 128), Q and the
full-resolution SW keys were rotated with `compress_rope_theta` (160000) + a
YaRN ramp instead of the main `rope_theta` (10000) with no YaRN.

The CPU reference (`reference.rs`, the test ground truth) does the opposite,
**uniformly for all layers**: Q (`reference.rs:307`) and `kv_sw`
(`reference.rs:318`) use `rope_cos` = `build_rope_cache(.., rope_theta)` (plain
RoPE, no YaRN — `build_rope_cache` at `reference.rs:1572` has no YaRN ramp).
ONLY the compressed keys use `rope_cos_c` = `build_rope_cache(.., compress_rope_theta)`
(`reference.rs:344`). The model intentionally dots a `rope_theta`-rotated Q
against `compress_rope_theta`-rotated compressed keys.

Why it stayed hidden until `seq > 128`: while every key lives in the SW window,
Q·SW-K is internally self-consistent under the (wrong but uniform) theta, so
output stays coherent. Once compressed / long-range keys join the softmax
(`seq > sliding_window`), the globally mis-rotated Q makes the attention
distribution collapse → repeat loop / no retrieval.

A second, lower-rank divergence: the compressed-key build
(`update_compressor_gpu_cache`) passed `original_max_position_embeddings`
(YaRN on) where the reference applies none.

## Fix

`infer/src/model/deepseek/weights.rs`:
- Q / SW-K / legacy output rope: always `(self.config.rope_theta, 0)`,
  independent of `compress_ratio` (matches the reference and the validated
  `forward_swa_attention_gpu` path).
- Compressed-key build: keep `compress_rope_theta`, set `original_seq_len = 0`
  (plain RoPE, no YaRN — matches `build_rope_cache`).

`crates/cuda-kernels/csrc/misc/dsv4_attention.cu`:
- CSA causal block count `available = abs_pos / ratio` (was `(abs_pos+1)/ratio`)
  to match the reference gate `block < t / ratio` exactly.

Validation: pod build + needle-in-haystack at `seq > 128` + CPU-reference
parity at ~200 tokens (covers SWA / CSA / HCA layers). **pending-remote** —
Mac has no nvcc (`cudarc` build.rs needs it), so compile + numeric validation
run on the 8×H20 pod; bench entry to follow under `wins/` once green.

## Rule

- RoPE base/scaling is per-TENSOR-ROLE (Q & SW-K & output = main theta;
  compressed-K = compressed theta), NOT per-layer-mode. Don't gate the Q/SW-K
  rope on `compress_ratio`.
- Validate long-context (seq > sliding_window) with an UNAMBIGUOUS probe
  (needle retrieval), not "looks coherent" — a model on a short attention
  window + priors emits fluent text that retrieves nothing. A "foundation
  correct" claim from short prompts does not cover the compressed path.
- When two paths (pool modes) behave identically, the bug is in shared code,
  not the diff.

## Validation update (2026-05-29, pod 8×H20)

Input-rope fix CONFIRMED CORRECT via the legacy path. With
`ARLE_DSV4_FLASHMLA_PREFILL=0 ARLE_DSV4_FLASHMLA_DECODE=0` (legacy
`dsv4_hybrid_attention_cuda`, which already applies the output inverse-rope),
the needle now retrieves at ALL lengths: prompt_tok 40 / 247 / 922 / 2272 all
HIT `ZORBLAX7` (was MISS at >128 before the fix).

But the DEFAULT FlashMLA paths regressed short context — an adversarial audit
found a SECOND, independent defect the input-rope fix does not cover: the
FlashMLA sparse **decode and prefill shims apply NO attention-output
inverse-rope** (reference.rs:417 `apply_partial_rope(dst, .., sign=-1.0)`),
which the legacy kernel does internally (dsv4_attention.cu:966-981). The
MODEL1 kernel (head_dim=512) keeps the rope tail in V, so the FlashMLA output
carries key-position rope that is never un-rotated at the query position →
collapse grows with |query_pos| (coherent ≤128, broken >128); changing the
input theta merely shifted which positions argmax survives, regressing short
context. Plus a third: the prefill HCA index builder
(`arle_flashmla_csa_prep.cu:174`) kept the `(abs_pos+1)/ratio` off-by-one the
decode/legacy fix removed.

Remaining fix (FlashMLA default paths): add the output inverse-rope (last
qk_rope_head_dim cols, sign=-1.0, rope_theta, no-YaRN, per-token
abs_pos=start_pos+token) after the decode fwd and the prefill fwd (incl. the
TP>1 sliced output), and change csa_prep.cu:174 to `abs_pos/ratio`.
