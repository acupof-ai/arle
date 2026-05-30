# Gated-delta-rule short-sequence prefill HANGS on H20 (seq_len < chunk span)

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

## Root Cause (HYPOTHESIS — not yet evidence-confirmed)

`ops::gated_delta_rule_prefill_chunkwise_batch_into` (or its preceding
`conv1d_prefill_packed_batch_into`) has a partial-chunk-only edge case that does
not terminate when `seq_len < chunk_size` (chunk span ≈32: 40 tokens works, 11
does not). Candidates, in order:
1. A TileLang gated-delta chunk kernel whose grid/loop is derived from
   `seq_len / chunk_size` (→ 0 full chunks for short seq) and spins or waits on
   an output a 0-chunk launch never produces.
2. The sm_90a recompile of the gated-delta cubin (the TVM target is still plain
   `sm_90` while nvcc now compiles `sm_90a`) miscompiles the partial-chunk WGMMA
   path. Weaker: arch mismatches are usually seq-length-independent, and the
   long-seq path works — but the partial-chunk path is a distinct code path.
3. `conv1d` causal prefill with `linear_conv_kernel_dim=4` on `seq_len < 4`
   (rules out 11-token, which still hangs — so not the sole cause).

License-or-kill: confirm by (a) re-probing the exact stuck kernel name with
`compute-sanitizer --tool synccheck` or finer probes inside the gated-delta op,
and (b) testing whether making the TileLang TVM target `sm_90a` (consistent with
the nvcc gencode) changes the behavior — that isolates hypothesis 2.

## Fix

OPEN. Workaround for validation: use prompts ≥40 tokens. The MoE e2e validation
stands (real-model coherent generation). A `guidellm` perf sweep is blocked until
this is fixed (default profiles use short prompts).

## Rule

A forward that passes a long-sequence warmup is NOT validated for short
sequences — the chunkwise/recurrent paths have seq-length-specific branches
(partial chunk, sub-kernel-dim conv) that a single warmup shape never exercises.
Smoke a new attention/prefill path across `seq_len ∈ {1, 2, < chunk, = chunk,
> chunk}` before declaring it working.
