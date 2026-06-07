# DSv4 batched decode: bit-identity is impossible by derivation — gate on coherence, not byte-parity

## Context

Step A of the unified batched-decode work (#38) lands a layer-major DSv4 decode
driver (`forward_decode_batch_stream_impl`) that runs N rows through each layer
together. The validation gate (`run_batch_decode_validate`) compared each batched
row to a c=1 single-row reference and **required byte-identity**:

- c=2: byte_parity=true (both rows == reference)
- c=3: byte_parity=false — row1 diverges at token idx2, row2 at idx6 (deterministic,
  reproduced across runs; full `ctx.sync()` between rows did not change it)

ckl's directive: **first DERIVE whether batched decode can ever produce zero
numerical difference vs single-row; don't obsess over exact values; then run a few
rounds and check whether the result is CORRECT (coherent), not byte-identical.**
(Aligns with `feedback_correct_inference_not_baseline_identity`.)

## The derivation — two op classes

Every per-layer op in the driver is one of:

**Class 1 — per-row single-column ops** (`mla_attention`, `dsv4_moe_forward`,
`dsv4_shared_expert_forward`). Row r is copied to a `[hidden,1]` scratch and run
through the *exact same kernel* the row-major reference uses, with the row's own
per-slot state (`slot.attention[layer]`, `slot.start_pos_device`) and its own KV
region (pool sliced by the slot's stored `slot_idx`). The kernel sees byte-identical
inputs → byte-identical output. **There is no numerical reason for Class-1 outputs
to differ across identical-input rows. A divergence here would be a logic bug.**

**Class 2 — batched ops over N** (`rms_norm_batch`, `hc_pre`/`hc_post`, `add_batch`,
`gen_mhc_params`, `all_reduce_sum`).
- The pointwise / per-token ops (hc, add, gen_mhc, and rms_norm's per-token
  reduction) compute `out[token r] = f(in[token r])` — no cross-row coupling, so
  they are bit-identical per row regardless of N.
- **`all_reduce_sum` over `[hidden, N]` is the one exception.** NCCL tiles a
  `[hidden,N]` message into per-rank chunks differently than it tiles N separate
  `[hidden,1]` messages. The cross-rank float accumulation an element lands in is
  message-shape-dependent, so identical-input rows pick up ~1 bf16 ULP of per-row
  drift. This is the same class of effect as MoE atomic-scatter non-determinism
  (`reference_dsv4_moe_nondeterminism_confounds_4096_parity`).

**Conclusion:** batched decode is **NOT** bit-identical to N single-row decodes,
and cannot be made so without doing N separate all-reduces (which defeats the
entire point of batching — one all-reduce is the throughput lever). This is
exactly the property of continuous batching in SGLang/vLLM: a request's output is
not bit-identical to running it alone, because the batched reductions round
differently depending on the rest of the batch. It is accepted as correct
inference everywhere. **Byte-parity-vs-c=1 is therefore the wrong gate.**

## Evidence (full-vector max-abs-diff probe, decode step 1, start_pos=5)

`INFER_DSV4_BATCH_PROBE=1` dumps per-stage max|row_r − row_0| over the WHOLE hidden
vector (elem0-only is blind — `hc_pre` mixes hc_mult lanes and `rms_norm` reduces
over the full vector, so any element's drift propagates):

| Stage (L0, SlidingWindow) | maxdiff vs row0 | meaning |
|---|---|---|
| `norm_in` (rms_norm out)  | **0.000000** | embedding + per-token rms_norm bit-identical |
| `attn_raw` (per-row MLA, **pre** all-reduce) | **0.000000** | Class-1 attention bit-identical ✓ |
| `attn_ar` (**post** all-reduce, pre hc_post)  | **0.031250** | ← FIRST divergence = the all-reduce |
| `attn` (post hc_post)     | 0.001953 | hc_post scales attn_out down vs the residual; seed already present |
| `moe` (end of L0)         | 0.014648 | drift compounding into L1 |

The first divergence is **not** at a Class-1 op (`attn_raw` = 0.000000, perfectly
identical) and **not** at hc_post (`attn_raw` was already 0 before it). It is the
`attn_raw → attn_ar` step — the **batched all-reduce** — that introduces it, then it
compounds through 43 layers until it flips an argmax at decode step 2 (idx2). The
c=2-vs-c=3 difference is explained: the all-reduce rounding pattern is N-dependent,
so a 2-column message and a 3-column message flip different tokens.

Noise floor: the c=1 reference re-run in-process is **bit-stable**
(`ref_self_parity=true`, `ref_self_first_div=None`). So this is NOT random MoE
non-determinism — it is the *deterministic, message-shape-dependent* batched
all-reduce numerics. Token results:

```
batch=1 reference         = [11111, 14, 778, 344, 990, 270, 6102, 294]
batch=1 reference rerun   = [11111, 14, 778, 344, 990, 270, 6102, 294]  ref_self_parity=true
batch=2 row0/row1         = identical to reference (first_div=None)
batch=3 row0              = identical to reference (first_div=None)
batch=3 row1              = [11111, 14, 260, 4593, ...]  first_div_vs_ref=2  (free-tail token flip)
batch=3 row2              = [11111, 14, 778, 344, 990, 270, 3924, 734]  first_div_vs_ref=6
```

All rows agree on the determined answer prefix `[11111("Paris"), 14(".")]`; divergence
is confined to the high-entropy free-continuation tail, the signature of legitimate
1-ULP numerics on greedy near-ties (`feedback_measured_floor`, degenerate-tail
sensitivity), not garbage.

## Correctness proof — needle retrieval (the right gate)

Byte-parity is wrong; the right gate is **determined-answer retrieval**. A 37-token
needle prompt embeds passcode "73914" (answer ids `[223, 30793, 929]`) and ends
"...The secret passcode is". Greedy decode, c=2/3/4:

```
reference = [223, 30793, 929, 16, 19018, 436, 7681, 16]   (retrieves 73914 ✓)
ref rerun = [223, 30793, 929, 16, 19018, 436, 7681, 16]   ref_self_parity=true
c=3 row0  = [223, 30793, 929, 16, 19018, ...]  first_div=None
c=3 row1  = [223, 30793, 929, 16, 3016,  ...]  first_div=Some(4)
c=3 row2  = [223, 30793, 929, 16, 3016,  ...]  first_div=Some(4)
```

**Every batched row retrieves the needle `[223,30793,929]` bit-identically** (plus the
following token); divergence is confined to idx≥4, the free-rambling tail. The harness
now gates on this: `INFER_DSV4_BATCH_MATCH_PREFIX=K` asserts every row matches the
verified-correct reference over the first K (= answer length) tokens. Byte-parity is
kept only as a reported metric, not an assertion (`INFER_DSV4_BATCH_STRICT` to restore).

## Rule

- **Bit-identity batched-vs-single-row is impossible by derivation** (all-reduce
  message-shape rounding). Never gate batched decode on byte-parity-vs-c=1. Gate on
  **coherence / needle retrieval / divergence-within-noise-floor** instead.
- The `attn_raw == 0.000000` evidence is the discriminator: per-row Class-1 compute
  IS bit-identical, so the batched path has no logic bug — the divergence is the
  legitimate batched-reduction numerics every continuous-batching engine exhibits.
- License-or-kill applies to the root-cause claim too: the full-vector probe (not
  elem0) + the post-all-reduce probe split + the ref self-parity rerun together
  pin the seed to the all-reduce with evidence, not inference.
