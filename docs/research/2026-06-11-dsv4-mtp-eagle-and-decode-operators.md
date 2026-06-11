# DSv4 decode research: DeepSeek 4-token MTP is EAGLE, and where our operators sit

**Date:** 2026-06-11. **Scope:** two ckl research asks — (1) why DeepSeek
"defaults to 4 tokens" for MTP and why ours collapsed at depth-4, (2) the
operator/算子 landscape for DSv4-Flash B=1 decode on 8×H20. **Backend:** CUDA
DSv4-Flash FP8 TP=8/EP=8. **Status:** research; revises the depth-K KILL verdict
(`errors/2026-06-11-dsv4-mtp-depth-k-draft-quality-wall.md`) from "head too weak"
to "linear-chain-limited — EAGLE tree is the licensed path."

## 1. DeepSeek 4-token MTP is one NextN head chained EAGLE-style — not 4 heads

**Config fact (pod `DeepSeek-V4-Flash/config.json`):** `num_nextn_predict_layers
= 1`. DSv4-Flash ships **one** MTP/NextN module — identical to DeepSeek-V3/R1.
The "4 tokens" is **not** four trained heads; it is the single NextN module
**chained 4 steps** under EAGLE-style speculative decoding.

**SGLang's mechanism (the reference impl):** DeepSeek MTP = EAGLE speculative
decode over the single NextN draft module. Multi-token comes from three knobs:
`--speculative-num-steps N` (chain depth), `--speculative-eagle-topk K` (tree
width per step), `--speculative-num-draft-tokens M` (tree nodes verified). The
NextN module's input is the EAGLE convention: `h' = e_proj(norm(embed(token)))
+ h_proj(norm(hidden_prev))`, attention over a **frozen** target KV, then the
shared output head samples the next draft.

**Measured acceptance (LMSYS, single NextN module):** average **2.44 accepted
tokens at 4-token MTP**, **2.18 at 3-token**; **1.8× decode at batch size 1**
on H200 TP8 (1.5× at bs=32). So the *one* head drafts ~2.44 tokens/step when
chained correctly.

**Our depth-4 measured 1/4 accepted** (drafts loop `[223,4489,223,4489]`,
2-cycle; pod e2e 2026-06-11). That is **3.4× worse acceptance than SGLang's
identical-architecture head.** The earlier KILL ("the 1-layer head can't draft a
chain, depth-K fundamentally dead") is therefore **mis-attributed** — the head
provably *can* draft 2.44 tokens. Our chain is broken.

### Why our linear chain collapses (root-cause hypotheses, ranked)

Our driver (`executor.rs` depth-K, `dsv4.rs::mtp_forward` +
`run_mtp_transformer_layer`) differs from EAGLE in three ways, each a candidate:

1. **Linear chain (topk=1), not a tree.** We draft one token per step and chain
   it; SGLang drafts a **tree** (eagle-topk>1) and verifies all paths, accepting
   the best. A linear chain commits to one wrong draft at step *i* and forfeits
   every later token — exactly the 1/4 plateau. This alone is most of the gap.
   *Strongest hypothesis.*
2. **Off-distribution hidden feedback.** We feed the single MTP layer's *own*
   output stream (`d_stream`) back as `h_prev` for the next draft. The module was
   trained as one of D *sequential* modules each consuming the prior's hidden, not
   one module consuming itself D times. SGLang hits the same off-distribution but
   the tree hedges it.
3. **Shared-KV chaining (needs code-trace confirm).** `run_mtp_transformer_layer`
   attends the *shared* frozen target KV pool at `start_pos+i`. Draft *i* writing
   its KV at `start_pos+i` into the target pool both (a) may not be visible to
   draft *i+1* depending on the read-window/position math, and (b) transiently
   corrupts the target KV the verify step reads (the DSA packed_rows self-heal,
   `dc068bc5`, addresses one facet of this). A rigid 2-cycle smells like draft
   *i* **not** seeing draft *i−1* — i.e. each draft re-derives "after 223 comes
   4489" from the *fixed* committed prefix, blind to the chain. Confirm by logging
   whether the draft attention length advances per chain step.

### Verdict revision

Depth-K is **not** dead. The licensed path to ~1.8× (from our current ~1.4×) is
**EAGLE-tree drafting + a dedicated draft KV cache**: (a) give the NextN draft
its **own** scratch KV (separate from the frozen target pool) so draft *i* attends
`[frozen target 0..start_pos] ++ [draft KV start_pos..i)` without corrupting the
target; (b) draft a **tree** (topk 2-4) and verify it with tree attention; (c)
keep the frozen-KV verify (no compressor re-run). The depth-1 clamp (`37986aeb`)
stays correct — *linear* depth-K is genuinely worse than depth-1 — but the wall
is the linear chain, not the head.

## 2. Decode operators (算子): inventory vs SGLang, ranked levers

**H20 reality:** ~39.5 ms/token wall (post-graph, parity shape). The 5–6 ms
target is an **H100/H800** number (where SGLang measures it); the H20
no-deep-rewrite floor is ~25–40 ms. Per-bucket
(`wins/2026-06-05-dsv4-decode-breakable-graph-launch-overlapped.md`): **hybrid
MLA attention ~11.8 ms = 30% of wall**, then FP8 GEMV, DeepGEMM pack/unpad, MHC.

### What we already have (rich)

`csrc/attention/`: FlashMLA decode (`dsv4_flashmla_decode_*`, **gated**),
`dsv4_swa_attention`, `dsv4_hybrid_attention`, `dsv4_csa_select`, FP8 KV pack.
`csrc/gemm/`: DeepGEMM native + bridge, `dsv4_grouped_gemm`, `moe_grouped_gemm`,
4 GEMV variants, Marlin W4A8. Runtime: fused-wqkv decode (gated, correctness-
licensed), comm-overlap (`dsv4_comm_overlap_enabled`, seq_len==1 non-deepep — a
Single-Batch-Overlap analog), DeepGEMM decode projections (default ON).

### The #1 operator lever: MLA weight absorption = the gated FlashMLA decode path

SGLang's headline MLA decode trick is **weight absorption** — fold `wkv_b`
(latent→K/V up-projection) into `wq_b`/`wo`, so decode attends **directly in the
~512-dim compressed latent** instead of decompressing to full per-head K/V. That
collapses the dominant attention bucket.

**Our default MLA decode does NOT absorb** — `attention.rs:2296` "all three modes
run through the bf16 correctness core … the perf-optimized FlashMLA sparse path
stays **gated**." So the 11.8 ms bucket is the **non-absorbed bf16 core**, and the
absorbed path (official FlashMLA decode, which operates on the latent ckv+k_pe) is
**gated-off**. Crucially, FlashMLA decode + fused-wqkv are **already correctness-
licensed** (`wins/2026-06-10-dsv4-lever-gate-license-or-kill.md`) — the default
flip is blocked only on a **wall-clock perf license**, not correctness. This is
the highest-leverage, lowest-risk operator move: A/B the gated FlashMLA decode
path against the bf16 core on the SLO shape and license-or-kill the 11.8 ms.

### Ranked operator levers (license-or-kill each on wall-clock)

1. **FlashMLA absorbed decode (the 11.8 ms / 30% bucket)** — gated, correctness-
   licensed; owes a wall-clock A/B. Single biggest decode lever.
2. **DeepGEMM pack/unpad fusion** — the FP8 quantize + layout ops wrapping the
   MoE/projection GEMMs are pure overhead; fuse into the GEMM epilogue/prologue
   (DeepGEMM groups only M; experts share N,K — fusion-friendly).
3. **Single Batch Overlap (H20-specific)** — SGLang modifies DeepEP+DeepGEMM to
   overlap dispatch/combine comm with expert compute. We have a comm-overlap path
   gated for the *allreduce* (non-deepep) transport; the deepep_ll lane does not
   yet overlap. H20 is comm-heavy at TP=8/EP=8, so this is real.
4. **MHC Sinkhorn fusion** — small bucket, overlapped; lowest priority (per the
   8 washes, only matters once the chain shortens).

**SOLID gate (owed before licensing any kernel lever):** a fresh decode-step
nsys with **per-kernel achieved-bandwidth + roofline%**, specifically the MLA
attention kernel — is 11.8 ms at the H20 bandwidth floor, or 2.5× above it? That
number, not commit-message verdicts, decides absorbed-FlashMLA's ceiling.

## 3. Synthesis: two multiplicative axes to the H20 floor

The 6 ms target is H100/H800; the H20 realistic floor is ~25–40 ms no-rewrite,
lower with the levers below. Two **independent, multiplicative** axes:

- **Amortization** — EAGLE-tree MTP: 1.4× (our depth-1) → ~1.8× (SGLang's 2.44
  tokens). Needs the dedicated draft KV + tree verify (§1).
- **Operator rewrite** — absorbed FlashMLA decode on the 11.8 ms MLA bucket
  (§2.1), then DeepGEMM fusion + SBO. Needs the nsys roofline license (§2 gate).

Earlier framing ("MTP is the only lever; kernels are dead") was wrong on both
counts — it conflated *compute-shaving of overlapped kernels* (genuinely dead)
with *operator rewrites that shorten the chain / absorb the attention* (wide
open), and it under-rated MTP by measuring a broken linear chain instead of the
EAGLE tree DeepSeek actually ships.

## Sources

- [LMSYS — Accelerating SGLang with MTP](https://www.lmsys.org/blog/2025-07-17-mtp/) (2.18/2.44 accepted tokens @ 3/4-token MTP)
- [SGLang DeepSeek-V3 usage](https://github.com/sgl-project/sglang/blob/main/docs/basic_usage/deepseek_v3.md) (single NextN module; EAGLE; backends supporting MTP)
- [LMSYS — Serving DeepSeek-R1 on H20-96G](https://www.lmsys.org/blog/2025-09-26-sglang-ant-group/) (H20-specific operator best practices, Single Batch Overlap)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) (M-grouped FP8 GEMM, expert-shaped)
- Internal: `wins/2026-06-05-dsv4-decode-breakable-graph-launch-overlapped.md`, `wins/2026-06-10-dsv4-lever-gate-license-or-kill.md`, `memory/reference_dsv4_decode_6ms_path_state`
