# Qwen3.5/3.6 MTP spec decode: rejection-sampling acceptance (RL-lossless sampled rollouts)

> Status: GPU gate RUN 2026-07-17 (H20, Qwen3.6-27B-FP8, depth=2, c=1) —
> correctness PASS, speedup 1.21× (below the 1.3-2× expectation; depth sweep +
> counters follow-up filed). Plan §P8
> (2026-07-16-agent-rl-unified-infra).

## Context

The checkpoint-native NextN-MTP lane was greedy-only (argmax-match verify) and
sampling-unreachable (`temperature != 0.0` fell back to plain decode), so RL
rollouts (temp=1) got zero spec-decode speedup. The sibling DSpark lane already
shipped the exact rejection machinery (`dspark_filter_probs_cuda` +
`dspark_chain_accept_cuda` + `dspark_draft_sample_cuda`); this is the mechanical
port onto the MTP lane — zero new CUDA.

## What Worked

- Draft levels sample on device from the engine-sampler-filtered head dist,
  retaining q per level; accept is chain rejection sampling (`u < min(1, p/q)`
  + residual `max(0, p−q)` draw). p and q share the SAME filter
  (temp/top_k/top_p/min_p), so committed tokens are exact w.r.t. the filtered
  target policy. Warm seeds sample instead of argmax.
- Identical rollback set to the greedy path: `restore_trunk` +
  `replay_linear_only` + `set_seq_len` (+ caller pool truncate); no new mutated
  buffer — sampled-mode scratch is read-only w.r.t. model/slot state.

## Gate results (2026-07-17, H20; MTP serve vs no-spec control, same binary)

- Needle ×3 at temp=1 (qwen3_nonthink RAW, lengths 115/446/1000/2000):
  **12/12 exact** on both arms.
- Same-config-twice (t=1, seed fixed, 200 tok): **bit-identical** on the spec
  serve (salted uniform streams); spec-on vs off same seed differ (expected,
  different sampler path), both coherent.
- Acceptance (via `ARLE_MTP_PHASE`; /v1/stats counters are a filed hole —
  executor.rs:369 maps only DSpark): sampled t=1 **50.9%** (394 chains,
  ≈2.02 committed tokens/spec step incl. bonus); greedy 61.5%.
- Decode tok/s A/B (t=1, 4 prompts × 512 tok, TTFT excluded): **52.56 vs 43.62
  = 1.21×**. Below expectation at depth=2/c=1; depth 3-4 sweep + decode-graph
  train-lane plumb filed as follow-up.
- Confound excluded first: tip garbled ALL temp>0 output on the no-spec control
  — single-variable attributed to b4b293f0c (hd256 q/k RMSNorm OFFSET→STANDARD,
  its wins entry was pending-remote/never GPU-run); the gate ran on tip+revert.
  That regression is owned by the hd256 lane and blocks any temp>0 use of
  Qwen3.6-27B-FP8 on tip until resolved.

## Rule

A spec-decode lane isn't RL-usable until sampled rows are reachable AND
exactness holds w.r.t. the filtered policy — port the shipped rejection twin,
don't invent a new acceptance rule.
