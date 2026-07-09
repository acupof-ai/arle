# DSpark temp>0 draft/accept device path (#13-②) — pending-remote

> Status: pending-remote — runtime license runs on the H20 pod.

## Context

Pod-measured: DSpark sampling mode inflated draft 16.6→71.7 ms (host per-row
filtered softmax over ~150K vocab × 16 markov steps) and accept_commit
2.0→18.7 ms (host p/q + residual sampling) → sampled spec 34.8 tok/s <
plain-sampling 37.6–37.8. Commits `e22a41637` (kernels+FFI) + `9f2dd5b3b`
(wiring) move both loops onto the device: per-markov-step
`dspark_draft_sample_cuda` (filter + q-row store + draw, 4-byte D2H) and one
`dspark_filter_probs_cuda` + `dspark_chain_accept_cuda` per verify (8-byte
D2H). Uniforms stay host salted splitmix64 `(seed, position)` streams.

## Pod gates (pending)

- sampled spec ≥ 37.8 tok/s (grep `[dspark-phase] ... draft= ... accept_commit=`;
  expect draft ≈ greedy + ~16 kernel syncs, accept_commit ≤ ~3 ms)
- same-seed-twice (cache off) byte-identical
- needle temp0.7 3/3
- greedy lane unchanged (byte-identical path — argmax branch untouched)

## Rule (provisional)

Host per-token vocab-wide loops in a spec-decode inner loop are a structural
tax; keep only token ids on the host and store full filtered dists in
pre-allocated device scratch.
