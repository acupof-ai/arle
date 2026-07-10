# DSv4 #150 MLA-proj bf16 both-lane lever — NULL on the concurrent truncation basin; batch-size dependence survives full projection-arithmetic unification

## Context

#150's residual mechanism hypothesis: concurrent digit corruption = near-tie
logit flips from batch-size-dependent arithmetic, localized (Part A layer-16
onset ordering) to the MLA gate `mla_attention_prepare_proj_batch`, whose
`wq_a`/`wq_b`/`wkv` are F8_E4M3-only in the checkpoint — n=1 decode ran M=1
fused FP8 DeepGEMM, n≥2 the M=N prefill-shaped FP8 DeepGEMM. The blocked
experiment (no bf16 weights exist on disk) was unblocked by shipping
`ARLE_DSV4_MLA_PROJ_BF16` (opt-in, default OFF): loader-side host FP8→bf16
block dequant copies of the three weights, and BOTH decode lanes (n=1 fused
AND n≥2 batched) routed through bf16 cublasLt so all decode batch sizes share
one arithmetic; prefill keeps FP8 DeepGEMM. Commits: `f261c6b03` (lever),
`047056fb4` (F32 power-of-two scale sidecar fix — the checkpoint's `.scale`
is F32, not E8M0; boot failed until normalized via `dsv4_block_scale_e8m0`
like the main FP8 load path).

## Experiment (pod, 8×H20, TP=4 GPUs 4-7, isolated tree `/host/arle-build-150` @ `095dcca6` = main+`047056fb4`)

Same binary all arms (`sha256 bac21cab…`, sccache-reproduced byte-identical),
same boot env (`ARLE_DSV4_MOE_BACKEND=allreduce ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_EXPERT_BACKEND=deepgemm --max-total-tokens 2048`),
`ARLE_DISABLE_PREFIX_CACHE=1` (isolate from reuse), fixed hard-salt carrier
`job2b-e5b-3` len 2000 (`needle150.py`, temperature=0, max_tokens=16; perf
lane max_tokens=128). Arms: C0 flags off · C1 `ARLE_DSV4_PROJ_BATCHED_BF16=1`
· C2 both flags. Per arm: solo sanity ×3, n=4 ×15 (60 req), n=2 ×20 (40 req),
solo ×15 (floor), perf n=4 ×15.

**Engagement proven** (flag-no-op trap closed): temporary once-`eprintln` in
both bf16 branches fired on all 4 TP ranks for BOTH lanes
(`[#150-probe] bf16 decode route engaged (n=1 lane)` ×4, `(n>=2 lane)` ×4);
probe reverted, verdict arms ran the clean binary. Load-side: 43 layers/rank
× +28.0 MiB = **1204 MiB/rank ledger; nvidia-smi 91303→92487 MiB (+1184)**;
load 50s→85s.

## Results (needle-miss rate; miss = `738291` absent)

| lane | C0 (off) | C1 (batched-bf16) | C2 (both) |
|---|---|---|---|
| solo n=1 ×15 | 46.7% (7/15) | 60.0% (9/15) | **53.3% (8/15)** |
| n=2 ×20 (40 req) | 82.5% (33/40) | 82.5% (33/40) | **75.0% (30/40)** |
| n=4 ×15 (60 req) | 91.7% (55/60) | 83.3% (50/60) | **85.0% (51/60)** |

First-boot arms (earlier binary `87e6f244`, flag-off byte-identical source)
reproduce the same picture: C0 88.3/77.5 (n=4/n=2), C1 85.0/77.5; C0 solo
46.7% vs C1 solo 86.7% — **boot-to-boot solo σ ≈ ±20pp** on this carrier, so
≤8pp arm deltas are noise.

Perf (C3, same-binary same-boot-shape): C2 vs C0 wall — n=4 mt16 9.554s vs
9.484s (+0.7%), n=2 4.855s vs 4.800s (+1.1%), solo 2.363s vs 2.369s (−0.3%),
n=4 mt128 9.620s vs 9.699s (−0.8%). All noise-level at these prefill-dominated
shapes (~10-16 decode tokens).

## Verdict

- **C2 ≈ C1 ≈ C0**: the sibling MLA FP8 gate is NOT the driver of this
  carrier's concurrent failure. Per the commissioning brief: report and stop.
- **The batch-size dependence survives full decode projection-arithmetic
  unification**: under C2 (engagement proven) n=1 misses 53.3% while n=2/n=4
  miss 75-85% in the same boot. Whatever still differs with n — top remaining
  candidate per the prior shortlist: the FP8 grouped-GEMM decode-MoE kernels
  (masked grouped GEMM shapes are token-count-dependent; never precision-A/B'd)
  — carries the divergence. Not chased this pass.
- **Signature caveat — digit-substitution was never sampled**: all ~690 scored
  requests across every arm/boot missed exclusively in the TRUNCATION class
  (`'…738.'` + EOS, incl. solo). Part A's layer-16 onset evidence implicated
  this gate for the DIGIT-SUBSTITUTION signature specifically; at HEAD on this
  carrier that signature produced zero events, so for digit-substitution the
  lever is **untested, not killed**. A carrier that reproduces `738292`-class
  flips at HEAD is a precondition for retesting.
- The lever stays as shipped: opt-in, default OFF, zero cost when unset
  (VRAM +1.2 GiB/rank and +35s load only when flagged on).

## Rule

**A "hard salt" carrier must be re-validated on the verdict binary before the
sweep is sized** — the e5b basin that missed 25-50% at n=4 (2026-07-09, cache
on/off) measured 47-92% at HEAD with a ±20pp boot-to-boot solo swing, i.e. the
carrier had drifted from "concurrency-triggered corruption" to "content basin
with concurrency amplification". Arm deltas smaller than the carrier's own
boot noise cannot license or kill anything; measure the floor's variance
(≥2 boots) before interpreting single-digit-pp deltas. And per the flag-no-op
rule: a NULL result requires positive engagement evidence (here the ×4-rank
once-eprintln probe), else "flag no-op passes exit-0" masquerades as a kill.
