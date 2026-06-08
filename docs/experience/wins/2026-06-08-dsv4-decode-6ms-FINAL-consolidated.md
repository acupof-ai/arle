# DSv4-Flash B=1 decode → 6ms: FINAL consolidated outcome (supersedes the 2026-06-07/08 trail)

This is the authoritative state of the decode-6ms investigation. It supersedes the
self-correcting trail of intermediate docs (host-bound → GPU-bound → per-kernel-dead);
those are kept for the lesson but the verdicts below are the final word.

## Delivered (committed, measured, byte-identical)

| metric | result | how |
|---|---|---|
| prefill | **23ms** | projections → DeepGEMM (prior work) |
| decode | **27 → 15ms** (+71%) | **MTP depth-1 batched-verify** (one 2-token forward amortizes the weight read; executor batched-reject = truncate + restore_spec_rollback + re-forward pending) |

`decode 15ms` is the realistic, sound B=1 ceiling on 8×H20 TP=8. **6ms is not achieved.**

## The diagnosis (definitive, measurement-backed)

**B=1 decode is critical-path-bound** — a fixed ~26ms/forward serial chain (43 layers ×
`hc_params → hc_pre → rms → attn → all-reduce → hc_post → moe → all-reduce`). The async
pipeline overlaps independent kernels, so **`gpukernsum` (GPU-time-per-kernel) ≠ the critical
path** — optimizing the biggest-GPU-time kernel overlaps away.

The active-weight floor is ~**0.3ms**/forward (13B active ÷ 8 GPUs ÷ ~4TB/s) — so the 26ms
forward is **~87× the floor**: the overhead is real engineering (M=1 latency-bound kernels +
the serial-chain dependency latency), NOT physics. **6ms is ~20× the floor → achievable**,
but only by changing the regime (amortization or chain-fusion), not point kernel tuning.

## Every lever, A/B'd (the evidence)

**Washed (wall-neutral — the wall is the critical path, these overlap):**
- per-layer CUDA graph (−5%); **whole-step CUDA graph** (86 ARs in one capture, byte-identical,
  WALL-NEUTRAL — the cleanest host-vs-GPU test → decode is GPU-bound, not host-bound);
- M=1 GEMV uint4 (1.8× isolated); mHC-fusion; **mhc_params uint4** (#1 by GPU-time, −25%
  isolated, wall A/B 38.95 vs 38.36 = wash); comm-overlap; alloc-pool; launch removal.
- **8 washes total → per-kernel/host/graph optimization is empirically DEAD for B=1 decode.**

**Killed:**
- DSA-skip (−3.7%, CSA compresses all tokens); depth-2 sequential MTP (~33% 2nd-token accept,
  only ~+15% — the single MTP head can't sustain multi-step drafts).

**The only lever that moved the wall:** MTP amortization (+71%). Because it spreads the WHOLE
fixed critical path over ~1.85 tokens/forward.

## Clean critical-path profile (per-forward GPU-time, rank-0 nsys, load separated)

ARLE-owned glue ≈ half: mhc_params 3.05ms (86×), cublas nvjet+splitKreduce ~1.8ms (~440× = the
3/4 un-fused MLA projections), pack_quantize 0.83ms (258×), rms 0.44ms. Vendored ≈ 4.4ms:
deep_gemm ~2.5ms, ncclAllReduce 1.3ms, flash_mla 0.64ms, get_mla_metadata 0.67ms. **But all of
these overlap** — reducing any individually washed (proven).

## The path to 6ms (all measurement-forced, none in-session-tractable)

6ms requires changing the M=1 critical-path regime, two routes:

1. **Amortization beyond depth-1** — tree-EAGLE (top-K candidates, one tree-masked verify).
   Est **~9-10ms** (SGLang EAGLE-2 class ~3 tok/fwd). Capped below 6ms by the model's **single
   MTP head** (`num_nextn_predict_layers=1`) — multi-step accept decays. Blocker: needs a
   tree-attention mask DSv4's FlashMLA (causal-only) lacks.
2. **Chain-shortening** — a fused mega-kernel that collapses the per-layer dependent ops into
   fewer GPU-side steps (removes inter-kernel dependency latency the whole-step graph can't,
   since the graph only pre-issues the same kernels). Major architectural (SGLang fused-decode).
3. **Batching (M=N, c>1)** — the throughput axis: amortizes the per-token critical path across
   requests → near-floor per-token latency at scale. The realistic 6ms-class route, but it's
   c>1, not strict B=1. (task #38, Codex worker credit-blocked until Jun 11.)

6ms for strict B=1 = tree-EAGLE **+** mega-kernel (confluence), at/below the practical B=1 floor.
For everyone on this model shape, ~15ms is a competitive B=1 single-request number.

## Gated infra retained (validated, default-off, zero behavior change)

- `ARLE_DSV4_WHOLE_STEP_GRAPH` — the validated whole-step capture (host-vs-GPU diagnostic +
  capturable-decode reference for the future fusion/tree work). Wall-neutral, so not default.

## Recommendation

15ms is the sound B=1 ceiling; the 6ms-class win is **batching (M=N)** — pivot there when the
throughput thread unblocks (Jun 11). Strict-B=1 6ms is a dedicated tree-EAGLE + mega-kernel
effort (multi-session, correctness-sensitive), now fully diagnosed and de-risked to its blockers
(tree-attention mask; serial-chain fusion). Per-kernel decode optimization is closed (8 washes).
