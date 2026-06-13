# #88 SGLang kernel-alignment validation — c=8 admission bistability kills a false GDN win; U3 OOM root-caused + fixed

**Date:** 2026-06-13. **Backend:** CUDA, Qwen3.6-35B-A3B, single H20 (96 GB),
TP=1, `--num-slots 8`. **Binary base:** 994c8f81 + lane-RUNS probes (isolates the
four #88 flags from the lead's concurrent NUMA-pin / DSv4-FP8 commits).
**Baseline (locked):** `q36_35b_head` 93.5 / 152.3 / 207.5 / 255.6 tok/s @
c=1/2/4/8 ([head-csweep](2026-06-12-qwen36-35b-head-csweep.md)).

## Context

The four Qwen-lane SGLang kernel swaps
([plan](../../plans/2026-06-12-qwen-lane-kernel-alignment-sglang.md), ckl
"kernel 全对齐 sglang") are each opt-in default-OFF: U2 GDN decode trio
(`ARLE_QWEN35_SGL_GDN`), U4 fused add-RMSNorm (`ARLE_QWEN35_FUSED_ADDNORM`),
U3 fused_moe (`ARLE_QWEN35_MOE_FUSED_SGLANG`), on the U1 triton-AOT infra.
Each needs the same gate before any default flip: lane actually RUNS (not a
no-op stub) → correct inference vs the off envelope → same-binary wall-clock
A/B clears per shape. This entry is the validation pass that settled all four;
the per-lane impl entries are [U2](2026-06-12-qwen35-sgl-gdn-triton-aot.md),
[U4](2026-06-12-qwen35-fused-addnorm-flashinfer.md),
[U3](2026-06-12-qwen35-moe-fused-sglang-u3.md).

## Verdicts

| Lane | Runs? | Correct? | Perf | Disposition |
|------|-------|----------|------|-------------|
| U1 triton AOT | — | — | — | **PASS** — all 7 cubins load non-stub (the 3 lane probes below fire, which only happens past the NOT_SUPPORTED loud-fail) |
| U2 GDN decode | yes (probe) | yes (= off math) | **WASH** | opt-in, no flip |
| U4 fused add-RMSNorm | yes (probe) | yes (DET, minor fusion-FP envelope caveat) | **WASH** | opt-in, no flip |
| U3 fused_moe | yes (probe) | n/a (OOM'd) | **OOM → fixed**, A/B pending-remote | opt-in, no flip |

## What worked — the c=8 admission bistability that manufactured a false win

A first cut (validate2, **n=3**) reported "**GDN +22% @ c=8**". It was an
artifact. High-rep re-run (**c=8 ×12**, same binary, off vs gdn) killed it: the
**off control itself is bimodal** — ~75% of c=8 reps land in a fast mode
(~258 tok/s, the value that scales smoothly from c=4's 207) and ~25% in a slow
mode (~210 tok/s ≈ the c=4 rate, i.e. no c=4→c=8 scaling at all).

Mechanism (kernel-INDEPENDENT, scheduler/arrival-timing): at `c == num_slots ==
8`, eight near-simultaneous arrivals are admitted as **either** one batch-of-8
(fast) **or** two waves-of-4 (slow, = c=4 throughput). Which one happens depends
on arrival jitter vs the admission tick, not on the decode kernel. Sampling n=3
on a 75/25 bimodal distribution routinely draws an off run from the slow mode and
a gdn run from the fast mode (or vice versa), fabricating a ±20% "effect" with no
kernel cause. The clean GDN signal is at c=1/2/4 (single admission wave, no
bistability): within ±~2.4% of off — a **wash**. U4 fused add-RMSNorm is likewise
DET-correct and a perf wash at every c.

This is the §0 framing-trap lesson in a new guise: a metric window (here a
single c=8 draw) that looks like a 22% win collapses under the conservative
framing (high-rep distribution). **License-or-kill must use the high-rep
distribution at `c == num_slots`, never a 3-sample mean.**

## DeepGEMM gives no benefit at the Qwen3.6 shape

`qwen35_deepgemm_enabled()` is default-ON and builds grouped-B caches at load
(clearing the per-expert Vecs). A/B at this shape: `dgoff`
(`ARLE_QWEN35_DEEPGEMM=0`, hand-grouped) ≈ `off` (DeepGEMM on) across all c.
DeepGEMM's FP8 / large-N tensor-core edge does not apply to per-expert
n=512 / k=2048 decode GEMMs — they are weight-read-bound, and DeepGEMM's JIT/TMA
fixed overhead on tiny routed bands is a wash-or-loss (consistent with the
in-tree hybrid-dispatch crossover note in `moe_forward_into`). Consequence for
U3: its control is `dgoff`, not `off` — both DeepGEMM-off so the A/B isolates the
fused-kernel swap rather than confounding it with the (no-op) DeepGEMM flip.

## U3 OOM — root cause + fix (this session)

The `moe` arm never served: `fused MoE w1 alloc failed: DriverError(
CUDA_ERROR_OUT_OF_MEMORY)` (pod, fresh). Root cause: the fused `w1 [E,2N,K]` +
`w2 [E,K,N]` BF16 cache is a **second full copy** of the MoE weights
(~1.6 GB/layer) and the first design built it **lazily on the first forward**,
ON TOP of the still-resident per-expert Vecs (DeepGEMM-off keeps them). On the
35B-A3B shard (~70 GB weights on a 96 GB H20, ~30 GB free), the per-layer lazy
builds exhausted the free pool around layer 20 (20 × 1.6 GB ≈ 32 GB).

Fix (`crates/infer-cuda/src/{loader,moe}.rs`): **build-and-replace at LOAD**,
mirroring the DeepGEMM grouped block in the same loader — restack the Vecs into
w1/w2, `ctx.sync()?`, then `gate/up/down.clear()` per layer. The lazy build
could not free the Vecs (it held `&MoeLayerWeights`, pool-shared `&self`); the
loader owns them `mut`. Resident is now net-zero growth (restacked ≈ freed);
peak = baseline + one layer's transient. `build_fused_sglang_weights` becomes a
pure cache getter; the forward gate loud-fails if the flag is on for a
non-device-route-eligible config (the non-fused fallback's Vecs are gone). Mac
typecheck + clippy green; the no-OOM confirm + needle-envelope + dgoff-vs-moe
A/B re-runs when the H20 frees (all 8 GPUs held by the lead's DSv4 TP=8/EP=8
serve, ~54 GB/GPU — the 35B BF16 needs a free single GPU). Detail in the
[U3 entry](2026-06-12-qwen35-moe-fused-sglang-u3.md).

## Rule

- **At `c == num_slots`, throughput is bimodal (one-wave vs two-wave admission);
  A/B it with the high-rep distribution, never a 3-sample mean.** A single c=8
  draw can fabricate a ±20% kernel "effect" that is pure admission luck. The
  clean kernel signal lives at `c < num_slots` (single admission wave). This is
  the same trap as the nsys-window framing: the narrow view (one draw / one NVTX
  window) lies; the conservative view (the distribution / wall-clock per request)
  is ground truth.
- **A second full-size weight cache must be build-and-replace at load, never
  lazy-on-first-forward.** A lazy build behind `&self` cannot free the source it
  duplicates, so it doubles resident VRAM and OOMs on a large shard. Mirror the
  loader's existing grouped-cache pattern (`build → ctx.sync()? → clear`) where
  the sources are owned `mut`. Enumerate the doubled buffer (here w1 1.07 GB +
  w2 0.54 GB per layer × 40 layers) before assuming "+1.5 GB total" — it was
  +1.5 GB *per layer*.
