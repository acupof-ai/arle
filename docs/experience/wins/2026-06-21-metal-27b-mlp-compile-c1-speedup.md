# Metal c=1 27B decode +51% — compile the MLP block (op-profile pinned it, NOT the gated-delta)

## Context
`arle serve --backend metal` at c=1 on Qwen3.6-27B-4bit (dense GatedDeltaNet) ran **9.7 tok/s**
vs raw `mlx_lm generate` **15.3** on the same model — a ~38 ms/step gap. Ruled out by measured
A/B: NOT the decode pipeline (off 9.66 ≈ on 9.75), NOT the KV dtype (int8 9.7 ≈ bf16 9.73).

## What Worked
**Op-level profile first (`INFER_METAL_OP_PROFILE`, eval-based per-section timer) — it overturned
my hypothesis.** I guessed the gated-delta state traffic; the profile showed the **MLP is the
dominant section (51%)**, gdr 30%, full-attn 10%, embed 5%, head 4%. The MLP's activation
(`compiled_swiglu`) was compiled but the **two matmuls + split were re-encoded per step ×32
layers** — the per-step graph-build/encode overhead, not compute.

**Fix:** `compiled_mlp_fn` — the whole MLP block (gate_up matmul → split → swiglu → down matmul)
as one cached compiled graph (`mlx_qwen35_model.cpp`). Keyed by (gate_dim, gs, bits) so one graph
serves every layer. **SHAPED not shapeless** (the split can't infer shapes shapeless) → gated to
**decode only (S=1)**, fixed shape → compiled once, reused; prefill keeps the per-op path.

Result: **9.7 → 14.63 tok/s (+51%)**, ≈ mlx_lm (96%). Correctness gate: **compiled vs
`INFER_METAL_NO_MLP_COMPILE=1` output is bit-identical** ("2, 3, 5, 7, 11, 13, 17, 19, 23, 29 …").

## Rule
- **Op-profile before optimizing a forward.** I twice nearly fixed the wrong thing (the agent's
  `eval()`-removal, my "compile is wrong / gated-delta") — the per-section eval-timer pinned the
  real cost (MLP) and the fix followed.
- **`mx::compile` of a matmul-heavy block is a real c=1 win even when the matmuls don't fuse** —
  the saving is the per-step graph-build/encode, not kernel fusion. Shaped compile + a fixed-shape
  (decode S=1) gate sidesteps the shapeless-split limit.
- A forward-path change ships only after **bit-identical greedy output vs the un-compiled path**
  (not just "looks coherent").
