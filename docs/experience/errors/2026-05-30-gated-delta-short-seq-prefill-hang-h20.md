# Gated-delta-rule short-sequence prefill HANGS on H20 (seq_len ≤ 32, chunkwise) — RESOLVED `e2246de1`

## Context

Serving `Qwen3.6-35B-A3B` (BF16) on one H20 (sm_90), `--disable-cuda-graph`,
num_slots=1. After the TileLang sm_90a fix unblocked the Hopper WGMMA attention
(see [`wins/2026-05-30-qwen36-moe-cuda-e2e-h20-real-model.md`](../wins/2026-05-30-qwen36-moe-cuda-e2e-h20-real-model.md)):

- 40-token prompt → coherent 24-token completion (prefill + decode OK).
- **11-token prompt (first request, fresh state) → HANGS** (GPU 100 %, request
  times out at 300 s, no error logged).
- **2-token prompt → HANGS** identically.

Localized with per-stage `eprintln!` probes under `CUDA_LAUNCH_BLOCKING=1`: the
2048-token warmup forward completes every layer (all 30 linear + 10 full attn +
40 MoE blocks print "done"); the short real request prints
`linear-attn enter linear_idx=0` and then **nothing** — it never reaches the
layer-0 MoE probe. So the hang is inside `prefill_linear_attention_paged_batch`
for layer 0, before MoE. `cuda-gdb -p` shows "No CUDA kernels" + host in `poll()`
→ the hung kernel is an external TileLang cubin (not introspectable), consistent
with the gated-delta chunk kernel.

NOT the MoE forward: the MoE kernels run correctly for the 2048-token warmup and
for the ≥40-token request (coherent output). Decode (single-step recurrent,
seq_len=1) also works. The failure is specific to the **chunkwise prefill path**
at small seq_len.

## Root Cause (CONFIRMED — runtime A/B, 2026-05-30)

The gated-delta prefill dispatch routes `seq_len <= 32` into the **chunkwise
TileLang pipeline** and `seq_len > 32` into the native, WGMMA-free **recurrent**
kernel — verified at `csrc/misc/gdr_prefill_batch.cu:137` (`if (seq_len > 32)
return gated_delta_rule_prefill_recurrent_cuda(...)`) and mirrored at
`infer/src/ops/recurrent.rs:686`. The chunkwise stages emit ten
`T.gemm(policy=GemmWarpPolicy.FullRow)` WGMMA calls
(`tools/tilelang/gated_delta_rule.py`). This maps 1:1 to every observation:
decode (seq_len=1, native fused) OK / `>32` (recurrent) OK / `<=32` (chunkwise)
HANG.

The culprit is the **chunkwise `FullRow`-WGMMA kernels themselves**, not the arch
flag and not the loader/metadata. Two independent pieces of evidence kill the
arch-split hypothesis (candidate 2 below):

- The full-attn kernel `batch_prefill_paged_hd256.py` **also** uses
  `GemmWarpPolicy.FullRow` WGMMA and runs **correctly on sm_90** (the MoE e2e).
  So `FullRow`-WGMMA works on sm_90 under the existing TVM-`sm_90` /
  nvcc-`sm_90a` split — the split does not universally break `FullRow`.
- A controlled A/B (below) reproduces the hang with **freshly-regenerated
  `sm_90a` cubins** (clean `8cd6c252` source + `gen_tilelang_aot.py` sm_90a fix +
  forced `tilelang_aot` regen), ruling out cubin staleness.

The fault is therefore a genuine TileLang **`FullRow` short-tile codegen bug** in
the gated-delta chunk kernels (the `solve_tril` / partial-chunk path at
`seq_len <= 32`), the same class as
[`errors/2026-05-27-tilelang-0110-fullrow-warp23-nan-sm80.md`](2026-05-27-tilelang-0110-fullrow-warp23-nan-sm80.md).
Candidates KILLED: (1) `num_chunks`/0-block edge — `num_chunks=1` for seq 2, 11
*and* 40, so it cannot discriminate; (2) arch split — see above; (3) `conv1d`
sub-kernel-dim — 11>4 still hangs and conv1d is shared by the working `>32` path.

## Fix (RESOLVED — `e2246de1`)

Env-gate `ARLE_GDR_CHUNKWISE_PREFILL`, **default OFF → route every `seq_len`
through the proven WGMMA-free recurrent kernel**. Gated in both the packed-batch
C path (`gdr_prefill_batch.cu` `gdr_chunkwise_prefill_enabled()`) and the
single-sequence Rust path (`recurrent.rs`), so the two paths agree (no
half-state). Chunkwise stays reachable via `ARLE_GDR_CHUNKWISE_PREFILL=1` on
arches where it is validated — which doubles as the root-cause A/B probe.

Runtime A/B on one H20 (sm_90), `--disable-cuda-graph`, num_slots=1, BF16,
freshly rebuilt from `8cd6c252` + the fix:

| `ARLE_GDR_CHUNKWISE_PREFILL` | path | 13-token prompt | GPU |
|---|---|---|---|
| OFF (default) | recurrent | **returns** (e2e essay prompt → coherent) | normal |
| `1` | chunkwise | **NO RESPONSE / HANG (50 s timeout)** | 100% util, wedged |

Fix verification sweep (flag OFF): `prompt_tokens ∈ {1, 13, 16, 20}` (all `<=32`,
all previously hung) → **all return, no hang**; the 20-token narrative prompt
emits coherent AI-history text via the recurrent path, which also re-validates
the Qwen3.6 MoE e2e from clean source. (Looping/short outputs on instruction-style
prompts under greedy decode are base-model artifacts, not a prefill-state bug.)

Deferred: restoring the chunkwise short-seq path on sm_90 needs the upstream
TileLang `FullRow` codegen bug fixed; tracked as a perf follow-up, not a
correctness blocker.

## Rule

A forward that passes a long-sequence warmup is NOT validated for short
sequences — the chunkwise/recurrent paths have seq-length-specific branches
(partial chunk, sub-kernel-dim conv) that a single warmup shape never exercises.
Smoke a new attention/prefill path across `seq_len ∈ {1, 2, < chunk, = chunk,
> chunk}` before declaring it working.
