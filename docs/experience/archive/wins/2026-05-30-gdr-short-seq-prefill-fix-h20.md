# Gated-delta short-seq prefill hang FIXED on H20 — env-gate chunkwise (default OFF → recurrent)

## Context

`Qwen3.6-35B-A3B` (BF16) on one H20 (sm_90) hung on every prompt with
`seq_len <= 32` (the gated-delta linear-attention prefill), blocking any
short-prompt serving and the default-shape `guidellm` sweep. Decode and
`>32`-token prefill worked (coherent), so the MoE e2e
([`2026-05-30-qwen36-moe-cuda-e2e-h20-real-model.md`](2026-05-30-qwen36-moe-cuda-e2e-h20-real-model.md))
stood but was un-servable for short inputs. Root-caused + fixed here; the error
entry is
[`errors/2026-05-30-gated-delta-short-seq-prefill-hang-h20.md`](../errors/2026-05-30-gated-delta-short-seq-prefill-hang-h20.md).

## What Worked

**Root cause (runtime-confirmed, not inferred).** The prefill dispatch routes
`seq_len <= 32` into the **chunkwise TileLang pipeline** (ten
`T.gemm(policy=FullRow)` WGMMA stages, `gated_delta_rule.py`) and `>32` into the
native WGMMA-free **recurrent** kernel (`gdr_prefill_batch.cu:137`,
`recurrent.rs:686`). The chunkwise `FullRow`-WGMMA kernels HANG on sm_90. The
arch-split hypothesis (TVM `sm_90` vs nvcc `sm_90a`) was **killed** by two facts:
the full-attn kernel `batch_prefill_paged_hd256.py` also uses `FullRow`-WGMMA and
runs fine on sm_90, and the hang reproduces with **freshly-regenerated `sm_90a`
cubins** (ruling out staleness). It is a genuine TileLang `FullRow` short-tile
codegen bug, same class as the 2026-05-27 sm_80 `FullRow` NaN.

**Fix (`e2246de1`).** Env-gate `ARLE_GDR_CHUNKWISE_PREFILL`, **default OFF →
route every `seq_len` through the proven recurrent kernel**, gated in both the
packed-batch C path and the single-sequence Rust path (no half-state). Chunkwise
stays reachable via the flag — which also serves as the root-cause A/B probe.

**A/B (one H20 sm_90, `--disable-cuda-graph`, num_slots=1, BF16, built from
`8cd6c252` + fix, forced `tilelang_aot` regen):**

| `ARLE_GDR_CHUNKWISE_PREFILL` | path | 13-token prompt | GPU |
|---|---|---|---|
| OFF (default) | recurrent | **returns** | normal |
| `1` | chunkwise | **NO RESPONSE / HANG (50 s)** | 100% util, wedged |

**Fix-verification sweep (flag OFF), regression gate per the error entry's Rule:**
`prompt_tokens ∈ {1, 13, 16, 20}` (all `<=32`, all previously hung) → **all
return, none hang**. The 20-token narrative prompt produced coherent AI-history
text via the recurrent path, re-validating the Qwen3.6 MoE e2e from clean source.
(Looping/short completions on instruction-style prompts under greedy decode are
base-model artifacts, not a prefill-state bug — the narrative prompt confirms the
forward state is correct.)

**Reproducibility note.** The H20 pod's on-disk tree had drifted back to a
Qwen3.6 `todo!` stub and its `target-pod/infer` was pre-Qwen3.6 — the prior e2e
binary was gone. Re-synced the pod to git-clean `origin/main` (`8cd6c252`) over
HTTPS (the pod rewrites GitHub through `ghfast.top`), applied the fix as a
verified patch (`git diff` == the two-file change), then rebuilt. Provenance is
git-clean per the 2026-05-28 pod-trust rule.

## Bench status

`pending-remote` for a full `guidellm` sweep: this is a **correctness** fix
(hang → no-hang), so the "before" is unmeasurable and the regression gate is the
seq-sweep above, not a throughput Δ. A `guidellm` short-prompt sweep is now
*unblocked* and is the natural perf follow-up — it would establish the first
Qwen3.6-CUDA-H20 short-prompt baseline (no prior baseline exists).

## Rule

A forward that passes a long-sequence warmup is NOT validated for short
sequences — the gated-delta dispatch branches on `seq_len` (`<=32` chunkwise vs
`>32` recurrent), and only the chunkwise branch carries the broken `FullRow`
WGMMA path. Smoke a new attention/prefill path across
`seq_len ∈ {1, 2, <chunk, =chunk, >chunk}` before declaring it working, and when
two kernels share a gemm policy but only one fails, the fault is the *kernel*, not
the policy or the arch flag — prove it by A/B with freshly-built cubins.
