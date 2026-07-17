# Qwen3.5/3.6 MTP spec decode: rejection-sampling acceptance (RL-lossless sampled rollouts)

> Status: pending-remote — GPU gate per plan §P8
> ([2026-07-16-agent-rl-unified-infra](../../plans/2026-07-16-agent-rl-unified-infra.md))

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

## Gate (pending-remote)

Needle ×3 + acceptance-rate + identical reward-curve-on-one-round per plan §P8,
plus same-config-twice sampling-distribution check. Expected 1.3-2× decode
speedup on temp=1 rollouts.

## Rule

A spec-decode lane isn't RL-usable until sampled rows are reachable AND
exactness holds w.r.t. the filtered policy — port the shipped rejection twin,
don't invent a new acceptance rule.
