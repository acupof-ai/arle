# gate_gap convention — machine-readable input-set provenance

Date: 2026-09-13. Proposed convention; two worked examples filled in
below, no retrofit of the existing gates.

## Why

The gate input-set audit
(`docs/plans/2026-09-13-gate-input-set-audit.md`) found the most common
honest verdict across all gates is "necessary but not sufficient": the
kernel arithmetic is tested at production geometry, but weights or width
are synthetic. That distinction is currently buried in prose per gate and
rediscovered by reading every example. Four fields make "does the input
set contain the production path" mechanically checkable.

## Fields (go on the registry row next to `correctness_gate` / `gate_gap`)

1. **`gate_shapes`** — the concrete dimensions the gate enumerates: the
   constant lists or sweeps (M/K/N, B, seq lengths, head counts, head_dim,
   page/block). Admits: a reader can tell which production sizes are
   exercised without opening the example. "sweep 1..256" or an explicit
   list; a single fixed value that omits a production width is the failure
   the marlin fixed-1024-column case exposes.
2. **`gate_dtype`** — operand and weight dtypes actually generated (bf16,
   fp8 e4m3 + e8m0 scales, int4 w4a16, fp8 KV, …) including whether the
   quantization is real (a production pack kernel) or an e4m3-magnitude
   stand-in. Admits: dtype agreement with the production tensor is visible.
3. **`gate_tp`** — the TP/EP/CP degree the gate instantiates (a list or
   "1 only"). A gate that proves a kernel at TP1 says nothing about the
   TP8 shard of a model that serves at TP8.
4. **`gate_weights`** — one of:
   - `rng` — deterministic synthetic weights (state the seed/vocab);
   - `checkpoint:<dir-or-env>` — loads a real model (state the path knob);
   - `derived` — tensors produced by the production pack/quant path from
     synthetic inputs (real transform, synthetic source).

## The derived boolean

`covers_production` = shapes include every production size the claim names
AND dtype matches AND gate_tp includes the serving degree AND
gate_weights is `checkpoint` or the operator is data-independent (an
argmax/sampler/elementwise/router-equivalence where RNG is a sound input).

The last clause is deliberate: requiring checkpoints for data-independent
kernels would add cost with no coverage. The convention must not label a
good synthetic gate as a gap.

## Worked example — honestly `yes`

`argmax_parity`:
- gate_shapes: batch × VOCABS `[7,128,1023,2048,2049,100000,151936]` (151936
  = production Qwen vocab)
- gate_dtype: f32 logits in, argmax index out
- gate_tp: n/a (pointwise)
- gate_weights: rng (SEED-keyed normal rows)
- covers_production: **yes** — argmax is weight/data-shape invariant; the
  production vocab width is in the sweep; deterministic RNG is the natural
  and sufficient input.

## Worked example — honestly `partial`

`dspark_draft_attn_parity`:
- gate_shapes: 40 q / 8 kv heads, head_dim 128, block 7 (matches
  Qwen3.8-27B-DSpark config); B=1 vs B=8, contexts 256..4096
- gate_dtype: bf16 dot vs f64 softmax, hash-filled synthetic KV
- gate_tp: 1
- gate_weights: rng (stable per-slot hash; no checkpoint)
- covers_production: **partial** — production geometry exactly, so kernel
  arithmetic and single-vs-batch equivalence are covered; real drafter
  weights and the multi-GPU path are not. A green gate here cannot speak to
  the 27B model-level acceptance collapse, and the `gate_gap` must say so
  in these terms.
