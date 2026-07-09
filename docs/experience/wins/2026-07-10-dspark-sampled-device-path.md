# DSpark temp>0 draft/accept device path (#13-②) — LICENSED, sampled spec 1.8–3.0× plain

## Context

Pod-measured: DSpark sampling mode inflated draft 16.6→71.7 ms (host per-row
filtered softmax over ~150K vocab × 16 markov steps) and accept_commit
2.0→18.7 ms (host p/q + residual sampling) → sampled spec 34.8 tok/s <
plain-sampling 37.6–37.8. Commits `e22a41637` (kernels+FFI) + `9f2dd5b3b`
(wiring) move both loops onto the device: per-markov-step
`dspark_draft_sample_cuda` (filter + q-row store + draw, 4-byte D2H) and one
`dspark_filter_probs_cuda` + `dspark_chain_accept_cuda` per verify (8-byte
D2H). Uniforms stay host salted splitmix64 `(seed, position)` streams.
License round: 8×H20 (GPUs 1/2/3), Qwen3.6-27B-FP8 + z-lab DFlash, binary =
tree at `0b5bd3d55` (see caveat below), temp 0.7 / top_p 0.95 / seed 42.

## Pod gates — all PASS

| gate | result |
|---|---|
| perf vs same-day plain-sampling anchor (35.5/36.3 tok/s) | csv **105.6 tok/s (2.97×)** accept 7.77; rust **64.2 (1.77×)** accept 3.75; step 67 ms |
| same-seed-twice, `ARLE_DISABLE_PREFIX_CACHE=1` | byte-identical (reasoning+content); diff-seed differs |
| needle 738291 ×3, temp 0.7, max_tokens 700 | exact 3/3 |
| greedy regression (cold) | csv 158.6 tok/s accept 8.70 @ 43.0 ms; rust 83.2 accept 3.11 @ 42.0 ms — 07-10 band endpoints exactly; sampling scratch +32 MiB lazy, greedy allocates none |
| OPD 3-turn shape (sampled, prefix-hit) | 77.1 → 68.3 → 62.1 tok/s; accept 4.15 → 3.54 → 3.15; turns 2–3 all `base>0` |

Phase means (sampled): draft 36.0 / verify 25.7 / accept_commit 4.3 ms.

**Next walls (measured, not hypothesized):**
- Sampled draft is 36 ms, not the projected 20–25 — the 16 per-step kernel
  syncs cost more than modeled; batching the sync or fusing markov-bias+sample
  is the next draft lever.
- Greedy prefix-hit accept drops harder than sampled (rust 3.11 cold → 1.92
  hit): partial-ctx accept cost is real on both lanes; the full-attention
  draft layer's blind span is the suspect.

**Provenance caveat:** committed HEAD `ce8c5dac1` did not compile for cuda —
`9f2dd5b3b` swept ckl's in-flight executor.rs/lib.rs call-site hunks (their
definition side landed in `0b5bd3d55` shortly after). The licensed binary
equals the `0b5bd3d55` tree. Hunk-split rule updated: pathspec limits files,
not hunks (memory `feedback_commit_only_own_files`).

## Rule

- Host per-token vocab-wide loops in a spec-decode inner loop are a structural
  tax; keep only token ids on the host and store full filtered dists in
  pre-allocated device scratch.
- Per-step device sync count is a first-class cost: 16 tiny kernels ≈ 20 ms of
  sync overhead here — model it before projecting kernel-move wins.
