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
| U3 fused_moe | yes (probe) | yes (needle env-match + c=8 coherent) | **KILL** −10/−33/−46% @ c=2/4/8 (no concurrency scaling) | opt-in, no flip |

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
mtime 07:59:42, 3 crates recompiled — non-stale). Detail in the
[U3 entry](2026-06-12-qwen35-moe-fused-sglang-u3.md).

### U3 validated — RUNS + correct, perf KILL (no concurrency scaling)

With both bugs fixed, the moe arm served and cleared correctness, then **lost
the perf A/B at every c**:

| c | dgoff (control) | moe (fused) | Δ% |
|---|-----------------|-------------|-----|
| 1 | 96.0 | 78.9 | **−17.8%** |
| 2 | 154.1 | 138.7 | **−10.0%** |
| 4 | 207.5 | 139.5 | **−32.8%** |
| 8 | 255.6 | 138.8 | **−45.7%** |

- **Lane RUNS:** moe probe fired (`SGLang fused_moe lane engaged top_k=8
  ep_size=1`); dgoff probe count=0. Not a no-op stub.
- **Correct:** RAW-needle envelope-match — moe = dgoff at len 115/446/2000
  (miss/partial/exact) and **strictly better at len 1000** (moe exact 3/3 where
  the hand-grouped control misses 3/3), all DET. Plus an explicit c≥2 check:
  the c=8 batched-decode responses are coherent English (essay text, well-formed
  `<think>`, no repetition/garbage). The fused kernel produces correct inference.
- **Perf KILL:** moe scales c=1→c=2 (+76%) then **plateaus flat** (138.7 →
  139.5 → 138.8 across c=2/4/8) — it saturates its grid at ~16 routed rows
  (c=2 × top_k=8) and serializes the rest, while dgoff keeps scaling 154 → 207 →
  256. SGLang's fused_moe is tuned for a different regime (large batch / EP /
  FP8); at the single-GPU bf16 Qwen3.6-A3B shape it caps at ~139 tok/s.
- **Measurement caveat (honest):** the c=1/c=2 Δ are same-run same-binary (v2).
  The c=4/c=8 v2 dgoff readings were `0.0` (failed requests) — a teardown-race
  artifact: the dgoff arm served *first*, seconds after 8× DSv4 CUDA contexts
  (~67 GB each) were SIGKILL'd, racing the kernel's context teardown; the moe
  arm ~10 min later was clean. The c=4/c=8 dgoff figures fall back to the
  triple-confirmed baseline (this-session v1 full sweep + locked 2026-06-12, both
  207.5/255.6). The KILL is certain regardless: moe's flat ~139 ceiling cannot
  approach dgoff's measured 207/256.

Disposition: **opt-in default-OFF, no flip.** Both load-path bugs fixed (the
lane is now usable for anyone who wants it), but it is a perf loss at this shape.

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
