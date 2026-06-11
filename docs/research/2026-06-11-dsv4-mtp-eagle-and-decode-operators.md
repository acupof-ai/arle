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

### Industry-leading DeepSeek MTP is a LINEAR chain (topk=1) — so our 1/4 is a BUG

**SGLang's default DeepSeek config: `--speculative-num-steps 3 --speculative-
eagle-topk 1 --speculative-num-draft-tokens 4`.** `eagle-topk 1` is a **linear
chain** (single branch/step) — *not* a tree. That linear 3-step chain reaches
2.18–2.44 accepted tokens. Tree (topk>1) is an *optional* further gain, and our
kernels can't do it anyway (FlashMLA decode is `s_q=1` scalar-only, no mask;
verify positions are contiguous-only — Explore confirmed). **So the target shape
is exactly our depth-K linear chain, and reaching it needs NO kernel build — just
fixing the chain bug that pins us at 1/4 vs SGLang's ~2.2.**

**Code facts (Explore):** the draft chain is *not* KV-blind — each `mtp_forward`
at `start_pos+i` writes its K into the shared pool and the next draft reads it
(SW ring / FP8 pool, same `target_attention_pool`). So the earlier "shared-KV
blindness" hypothesis is **falsified**. The rigid `[223,4489]` 2-cycle is a chain
bug, candidates (need a pod probe to isolate — one variable at a time per §0):
1. **Off-distribution hidden feedback** — we feed the single MTP layer's *own*
   wide stream (`d_stream`) back as `h_prev`. SGLang's linear chain feeds the
   draft's hidden too and works, so either our `d_stream` is the wrong tensor/
   slice, or it needs the same norm SGLang applies. *Prime suspect.*
2. **Position / RoPE on the chained draft** — draft *i* at `start_pos+i`; if the
   RoPE position or the device `start_pos` is mis-fed on chain steps >0, the draft
   attends with wrong positional encoding → degenerate fixed point.
3. **Draft KV corrupting the committed pool** — drafts write "future" slots
   `start_pos..start_pos+depth` in the shared pool; on the chain this is the
   intended self-attention, but verify/rollback must overwrite/restore them
   cleanly (the DSA packed_rows self-heal `dc068bc5` covers one facet).

### Verdict revision

Depth-K is **not** dead and does **not** need a tree kernel. The path to SGLang-
class (2.18–2.44 tok, ~1.8×) is to **debug our linear chain** to parity with
SGLang's linear `topk=1` chain. The depth-1 clamp (`37986aeb`) stays correct as a
safety until the chain is fixed; once per-position accept clears ≥2 tokens at
depth-3, unclamp. Diagnose first (§0: isolate the 2-cycle's single cause on the
pod), then fix — do **not** charge into the tree (topk>1) build, which is a later,
kernel-deep, optional gain.

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

### CORRECTION (Explore, 2026-06-11): the big operator levers are ALREADY landed

My first draft said "our default MLA decode does NOT absorb (FlashMLA gated-off)"
— that read a **stale comment** (`attention.rs:2296`). The actual gate
`dsv4_flashmla_decode_enabled()` (`attention.rs:2755`) is **default-ON**:
> "Default ON: FlashMLA SM90 sparse decode is the adopted decode attention — the
> same vendored kernel SGLang uses. Licensed 2026-06-06: 29.47 → 36.59 tok/s
> (+24%). Opt out with `ARLE_DSV4_FLASHMLA_DECODE=0`."

So the **MLA weight-absorption lever is already shipped and default-on (+24%)**,
alongside **fused-WQKV decode** (`attention.rs:2949`, default-on, +18.4%,
2026-06-06) and **DeepGEMM decode projections** (default-on, +6.3%). The 11.8 ms
MLA bucket from the 2026-06-05 entry **predates** the 2026-06-06 FlashMLA-default
flip — the current decode already runs the absorbed FlashMLA path. **"算子做好"
is largely DONE.**

Remaining operator headroom is incremental, gated on a fresh nsys roofline
showing the current FlashMLA decode is *above* the H20 floor:
1. **nsys roofline%** on the current FlashMLA decode kernel — is it at the H20
   bandwidth floor, or is there headroom? (Decides whether anything below is worth it.)
2. **DeepGEMM pack/unpad fusion** — FP8 quant/layout overhead around the MoE/proj GEMMs.
3. **Single Batch Overlap for the deepep_ll lane** — we have comm-overlap for the
   *allreduce* transport (`dsv4_comm_overlap_enabled`, seq_len==1, non-deepep) but
   not the deepep_ll path; SGLang overlaps DeepEP dispatch/combine with DeepGEMM.

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

The 6 ms target is H100/H800; the H20 realistic floor is ~25–40 ms no-rewrite.
Two **independent** axes, and after the Explore both are clearer:

- **Amortization (the open work)** — fix our **linear** depth-K chain to SGLang's
  linear `topk=1` parity: 1.4× (depth-1) → ~1.8× (2.18–2.44 tok). No kernel build;
  it's a chain bug (§1). This is where "industry-leading MTP" actually lives.
- **Operator (largely landed)** — the absorbed FlashMLA decode (+24%), fused-WQKV
  (+18.4%), DeepGEMM projections (+6.3%) are **already default-on**. Remaining is
  incremental (DeepGEMM pack/unpad fusion, SBO for deepep_ll), gated on an nsys
  roofline showing the current FlashMLA decode is above the H20 floor (§2).

Net: the headline lever is the **MTP linear-chain fix** (toward 2.2 tok); the
operator side is mostly done and its remainder is gated on a roofline measurement.

## Sources

- [LMSYS — Accelerating SGLang with MTP](https://www.lmsys.org/blog/2025-07-17-mtp/) (2.18/2.44 accepted tokens @ 3/4-token MTP)
- [SGLang DeepSeek-V3 usage](https://github.com/sgl-project/sglang/blob/main/docs/basic_usage/deepseek_v3.md) (single NextN module; EAGLE; backends supporting MTP)
- [LMSYS — Serving DeepSeek-R1 on H20-96G](https://www.lmsys.org/blog/2025-09-26-sglang-ant-group/) (H20-specific operator best practices, Single Batch Overlap)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) (M-grouped FP8 GEMM, expert-shaped)
- Internal: `wins/2026-06-05-dsv4-decode-breakable-graph-launch-overlapped.md`, `wins/2026-06-10-dsv4-lever-gate-license-or-kill.md`, `memory/reference_dsv4_decode_6ms_path_state`
