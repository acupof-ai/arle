# GDR projection compile — KILL (the gated-delta cost is the custom kernel, not the projection encode)

## Context
After `compiled_mlp_fn` won +51% on the Metal 27B decode (9.7→14.6 tok/s, matmul-encode bound),
the re-profile showed GDR (gated-delta, 24 layers) as the next-biggest quality-neutral section
(~27%). Hypothesis: the GDR projection matmuls re-encode per step ×24, same overhead the MLP
compile removed. Tried `compiled_gdr_in_proj_fn` + `compiled_gdr_out_proj_fn` (shaped, decode S=1,
gated, mixed-bit keyed), leaving the conv / custom `fast::metal_kernel` delta-rule / state untouched.

## Root Cause (of the KILL)
**The GDR wall-clock cost is the custom delta-rule Metal kernel (recurrent compute), NOT the
projection encode.** Matched A/B on OptiQ (`mlx-community/Qwen3.6-27B-OptiQ-4bit`): compile ON
**12.52** == `INFER_METAL_NO_MLP_COMPILE=1` OFF **12.52** tok/s, Δ=0, output bit-identical
(sha256 match). A stderr probe confirmed the compiled path *executes* — it just yields zero
wall-clock. Unlike the MLP (2 big matmuls + split, matmul-dominated), the GDR projections are a
small per-layer share; saving their re-encode doesn't move the wall at ×24 layers. The MLP win
does NOT generalize — "compile the matmul-heavy block" only pays where the matmuls dominate.

## The confound (the real §0 lesson)
The first ON measurement read **9.33 tok/s — a false regression**. Cause: that serve booted with
31.4 GiB available vs 33.9 for the OFF arm, so the Metal resource guard **clamped KV to 1904
tokens** (vs 21232) and the process thrashed. A fresh `sudo purge` to match available memory
(33.9 GiB, KV 21232) gave 12.52 = OFF. On the 48 GiB box, OptiQ's ~19 GiB weights sit close to the
guard threshold; **any A/B must `sudo purge` and confirm matched `kv_capacity_tokens` first**, or
memory pressure — not the code — moves the number.

## Rule
- **The encode-bound win is block-specific.** Profile the actual section before assuming the MLP
  compile transfers; a section that's custom-kernel / recurrent-compute bound won't respond to
  projection-fusion. KILL on Δ≤0.3 and revert clean — no dead gated code.
- **Match Metal memory before any big-model A/B.** `sudo purge` + assert equal
  `kv_capacity_tokens` across arms; an unmatched resource-guard KV clamp fabricates regressions
  (here 9.33 vs the true 12.52). Pairs with [[feedback_matched_ab_for_small_bench_effects]] and
  [[feedback_vram_accounting_bit_exact]].
- OptiQ Metal c=1 floor is **~12.4 tok/s** (PPL 7.82, vs plain 4bit 14.6 / PPL 8.56): the MLP
  compile was the one big lever; the 8-bit MLP ups + 8-bit lm_head + the GDR kernel are
  quality/compute-bound, not optimizable without losing precision.
