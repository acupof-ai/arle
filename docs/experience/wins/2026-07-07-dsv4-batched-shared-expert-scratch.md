# DSv4 batched-decode shared-expert scratch reuse — correctness verified, alloc removed

> Status: Verified (correctness) | wall-clock A/B deferred | 2026-07-07

## Context
Launch-bound Step 1 (`docs/plans/2026-07-07-dsv4-decode-launch-bound-plan.md`).
nsys `kern141_decode2` (07-03, TP=4, MTP-on): DSv4 decode is launch-bound —
`cudaLaunchKernel` 39.8% + `cuStreamSynchronize` 26.6% = 66% wall, zero
`cuGraphLaunch`; `cuMemAllocAsync`+`Free` 12.2M calls (7.7%) + `cuMemsetD8Async`
2.4M (9.1%) = 16.8% wall of per-step device allocation. Commit `de6fc4fd`.

## What Worked
`forward_decode_batch_stream_impl` (the live MTP/N>1 decode path) allocated a
fresh shared-expert output (`HiddenStates::uninit`) + the ~6 FP8 scratch allocs
+ 4 tiny H2D inside `dsv4_shared_expert` **every layer every step**. Switched it
to the model-wide `Dsv4SharedDecodeScratch` already held on the kv_adapter
(#29, `kv_layout.rs:159/494`, allocated whenever the model has a shared-expert
layer — independent of decode graph) via `dsv4_shared_expert_forward_decode_scratch`,
mirroring the eager/verify template (`dsv4.rs:5440`).

- **Compile**: BUILD_EXIT=0, borrow-check clean (the MoE-half
  `kv_adapter.shared_expert_decode_mut()` borrow does not overlap the attention-half
  `layer_dsa_and_flashmla_batch_mut` borrows — they run sequentially per layer).
- **Correctness (§0 case-as-fact)**: TP=4/EP=4 on GPU 4-7, DeepSeek-V4-Flash-FP8,
  greedy. Two decoded prompts returned **coherent** reasoning
  ("...The answer is straightforward: Paris.", "...capture the essence of the
  sea..."). No garbage, no NaN, no empty generation. The `needle_gate.py` run
  reported all-miss `out=''` — that is a **harness artifact** (DSv4 puts the
  answer in `reasoning_content`/think, `content` empty at `max_tokens=24`,
  `finish_reason:length`), NOT a regression. Same class as the prep-norm
  "harness-truncated miss" and the AGENTS.md §0 anchor (decode the tokens; the
  aggregate metric lied).
- **Alloc removal (code-certain)**: removes 1 `uninit(hidden, n)` + `dsv4_shared_expert`'s
  6 allocs + 4 H2D per layer per step on the batched path — the memsets among them
  (`zeros`/`alloc_zeros`) were part of the 9.1% `cuMemsetD8Async`.

## Honest scope
This is **one** of the ~45 per-layer allocs (the shared-expert cluster). Alloc
COUNT drops by construction; **wall-clock impact of this commit alone is expected
small** (the shared-expert allocs are a fraction of the 16.8% alloc wall). The
real launch-bound win requires the full Step-1 sweep (commit 2: the ~8
`dsv4_moe_forward_decode_fp8` allocs; then the attn/ffn stream buffers + N-ring
prepared buffers). Deferring the wall-clock A/B until the allocs are removed as a
batch — a per-commit A/B would measure noise. Correctness is gated per commit
(done here); wall-clock is gated on the accumulated Step-1 diff.

## Rule
On a launch-bound path, a single alloc-site removal is correctness-gated per
commit but wall-clock-gated on the accumulated batch — one site is noise-level,
the sweep is the lever. Do NOT claim a per-commit wall-clock win; do verify
per-commit correctness (decode real tokens, not the needle aggregate).
