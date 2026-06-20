# DSv4-Flash B=1 perf — candidate optimization queue (backlog, draft)

Living backlog. **Each item is a hypothesis, not a commitment** — per §0 every entry
must be decomposed to file:line + measured (nsys fraction / same-binary A-B) BEFORE
any code, and cost is stated only AFTER the decomposition (per `feedback_no_ungrounded_estimates`).

## State anchor (what's measured so far)

- B=1 decode is **foundation-bound**, NOT host-launch-bound (verified: per-step
  `ctx.sync` at `ops.rs:467` → host can't run ahead; the graph −41% was the control).
  See `errors/2026-06-20-host-launch-bound-misinference-decode-is-foundation-bound.md`.
- My earlier "csa_select = 70%" was measured on the **legacy fallback** path — a
  footgun now DELETED (`16a4ada2`, B=1 routed to the official DSA select). **The real
  official-path B=1 bottleneck is being re-measured on pod** (sync→build→serve→needle+tok/s recipe).
  Update this queue with the real breakdown when it lands.
- **The full per-worker decode CUDA graph is dead** (3 kill records): it can't remove
  the per-step `ctx.sync` (host already blocks) nor the cross-process per-tick lockstep
  barrier (`serve_multiproc.rs`, 4 OS processes), and adds replay tax. Any graph candidate
  below must explicitly avoid those two.

## Candidates (ckl 2026-06-20 + the levers from the foundation-bound analysis)

1. **Draft-model (MTP draft head) kernel graph-ization.** The MTP draft chain
   (`mtp_forward_level`, `executor/spec_decode.rs` draft loop) launches its own kernels
   per draft step. Graph-capture the draft head's kernel sequence where it does NOT cross
   the per-tick cross-process barrier (the draft runs within one verify step). Gate: nsys
   the draft phase (currently ~4.4ms of the MTP step) — is it launch-bound within the step?

2. **MTP-path kernel graph-ization.** Same idea for the MTP verify/commit kernels
   (`spec_step`, `commit_accepted_fold`). The verify is the 75% of the MTP step; if its
   intra-step kernel launches (not the comm) are graphable, capture them. Gate: nsys the
   verify forward's launch-vs-compute split.

3. **PARTIAL kernel graph-ization (the key nuance).** The full decode graph regressed
   because it tried to capture the whole step incl. the cross-process barrier + couldn't
   skip the `ctx.sync`. A PARTIAL sub-graph of a kernel-heavy, comm-free, sync-free
   *portion* (e.g. the per-layer projection-GEMV cluster, or the attention-math kernels
   between two AllReduces) might still cut launches without hitting the killers. Gate:
   isolate a contiguous comm-free kernel run in the nsys timeline; A-B a sub-graph of it.

4. **Reduce syncStreaming / comm serialization.** Two concrete sites:
   - `ops.rs:467` per-step `ctx.sync` (the host barrier every token) → device-side
     sampling (token stays on device, position by-reference) so the host can run ahead —
     this is also the prerequisite that would let any graph (1/2/3) actually pipeline.
   - `tp.rs:297` AllReduce "runs in place on the compute stream … no cross-stream event" →
     there is currently NO comm/compute overlap. A separate comm stream + events could
     overlap the per-layer AllReduce with the next layer's compute. Gate: nsys the AllReduce
     kernel fraction (it was ~3% on the legacy-path trace — re-measure on the official path;
     only worth it if the official-path comm fraction is materially larger).

5. **Kernel details under-optimized — dig the real bottleneck.** "一些算子细节做的还不充分."
   Once the pipeline's nsys names the real official-path B=1 top kernel(s), decompose each
   to its launch config + memory/compute floor (like the csa_select 1-block finding). The
   M=1 GEMVs (`gemv_handwritten` was 11% on the legacy trace), the DSA official select
   (DeepGEMM paged-MQA topk), the Sinkhorn `mhc_params` — each gets a floor check.

6. **Code quality: 拆分复用 (decomposition + reuse) as a standing bar.** Every new op /
   refactor: split into reusable units, converge to one canonical path, delete cruft (cf.
   the `dsv4_csa_select` legacy delete + `forward_mtp_warm_step` reuse). No layered adapters,
   no parallel old+new paths.

7. **TP/EP sharding rethink — TP only where needed, replicate small layers, load-balance
   (ckl 2026-06-20).** Hypothesis: instead of TP-sharding every layer (each incurring a
   per-layer AllReduce — part of the foundation wall), TP-shard ONLY attention; replicate
   the small dense compute layers (norms, router, shared expert, MLA LoRA projections) on
   every GPU + load-balance compute → fewer AllReduces/layer → lower per-token comm +
   barrier cost. Single-node (8 GPU) makes replication cheap. Gate: map the current ARLE
   DSv4 sharding (what's TP vs EP vs replicated, the AllReduce/AllGather count per layer)
   → nsys the comm fraction on the OFFICIAL path → A-B the replicate-small-layers variant.
   Connects to the foundation-bound finding + single-process-TP. **Subagent investigating.**

## Process

Pull from this queue only AFTER the pipeline re-nsys names the real bottleneck — pick the
candidate that the measurement says owns the wall, decompose it, A-B it via the canonical
`scripts/bench_guidellm.sh` (or the pod sync→build→serve→profile recipe). Do not implement on inference.
