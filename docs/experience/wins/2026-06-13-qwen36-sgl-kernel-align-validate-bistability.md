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
| U3 fused_moe | yes (probe) | n/a (didn't serve) | **2 load-path bugs → both fixed**, A/B pending-remote | opt-in, no flip |

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

## U3 — two load-path bugs, both found by validation + fixed (this session)

The `moe` arm failed to serve **twice**, for two distinct reasons surfaced in
sequence by the same validation harness. Both are now fixed; the perf A/B is
the only step left (GPU-blocked).

### Bug 1 — OOM (memory doubling)

`fused MoE w1 alloc failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY)` (pod,
fresh). Root cause: the fused `w1 [E,2N,K]` +
`w2 [E,K,N]` BF16 cache is a **second full copy** of the MoE weights
(~1.6 GB/layer) and the first design built it **lazily on the first forward**,
ON TOP of the still-resident per-expert Vecs (DeepGEMM-off keeps them). On the
35B-A3B shard (~70 GB weights on a 96 GB H20, ~30 GB free), the per-layer lazy
builds exhausted the free pool around layer 20 (20 × 1.6 GB ≈ 32 GB).

Fix (`59cea517`, `crates/infer-cuda/src/{loader,moe}.rs`): **build-and-replace
at LOAD**, mirroring the DeepGEMM grouped block in the same loader — restack the
Vecs into w1/w2, `ctx.sync()?`, then `gate/up/down.clear()` per layer. The lazy
build could not free the Vecs (it held `&MoeLayerWeights`, pool-shared `&self`);
the loader owns them `mut`. Resident is now net-zero growth (restacked ≈ freed);
peak = baseline + one layer's transient. `build_fused_sglang_weights` becomes a
pure cache getter; the forward gate loud-fails if the flag is on for a
non-device-route-eligible config (the non-fused fallback's Vecs are gone).

### Bug 2 — expert-count validator rejects the cleared-Vecs form

With Bug 1 fixed, the re-run's `moe` arm **still** failed — but no longer OOM:
`engine step failed: MoE expert count mismatch: gate=0 up=0 down=0
local_experts=256 (ep_size=1 ep_rank=0)`. Bug 1's fix had done its job (the
Vecs were freed), but `moe_forward_into`'s early `!use_deepgemm` precheck only
accepted two weight forms — `gate_grouped.is_some()` (DeepGEMM) or full
per-expert Vecs — and tripped on the freed Vecs **before** control reached the
fused branch (which returns at `moe.rs:813`, well past the check). Fix
(`ad39dc77`): add `|| weights.fused_sglang.is_some()` as the third accepted
form, mirroring how `gate_grouped` is already accepted. The fused branch and
`add_shared_expert_gated` (shared-expert weights, not the routed Vecs) were the
only other consumers — enumerated, both safe. **Codex review (`--commit
59cea517`) independently flagged this exact precheck gap and prescribed the same
fix** ("Include the fused cache in that precheck") — static analysis and the pod
failure converged on the same root cause.

Mac typecheck + clippy green for both fixes; pod tree staged + rebuilt (binary
mtime 07:59:42, 3 crates recompiled — non-stale). The no-OOM confirm +
needle-envelope + dgoff-vs-moe A/B re-runs when the H20 frees (all 8 GPUs held
by the lead's DSv4 TP=8/EP=8 serve, ~55 GB/GPU at 100% util — the 35B BF16 needs
a free single GPU; per "等等 h20 的使用" no GPU is grabbed mid-run). Detail in the
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
- **A load path that frees the per-expert Vecs must be allowed past EVERY
  Vec-length precheck, not just the kernel it feeds.** Build-and-replace clears
  `gate/up/down`, so any forward-side validator asserting `gate.len() ==
  local_experts` rejects the new weight form before the consuming branch runs —
  even when that branch never touches the Vecs. Grep every reader of the cleared
  buffer (`weights.{gate,up,down}` len/index/iter) and prove each: here the
  precheck (`moe.rs:628`) needed the third OR arm; the shape-resolution block
  (`moe.rs:919`) was already past the fused early-return; `add_shared_expert_gated`
  reads shared-expert weights, not the routed Vecs. Codex review caught the same
  gap by static analysis — the §0.1 "enumerate every mutated buffer + prove each"
  discipline applies symmetrically to every *reader* of a freed buffer.
