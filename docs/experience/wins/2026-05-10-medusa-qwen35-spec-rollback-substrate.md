# Qwen3.5 Medusa Spec Rollback Substrate

Status: pending full Medusa bench

## Goal

Land the Qwen3.5 target-state rollback substrate needed by Medusa Phase 1.B:
paged KV rollback remains scheduler-owned, while Qwen3.5 restores its recurrent
state from the verifier snapshot-ring slot matching the accepted draft length.

## Hypothesis

Qwen3.5 speculative verification can be made correctness-safe by saving
recurrent snapshots after each verifier row and restoring slot `num_accepted`
after greedy verification, aligning recurrent state with
`paged_kv_pool.truncate_slot(original_len + 1 + num_accepted)`.

This uses the CUDA Graph snapshot-ring path licensed in
[`2026-05-10-medusa-qwen35-snapshot-ring-graph-license.md`](2026-05-10-medusa-qwen35-snapshot-ring-graph-license.md):
K+1=6 snapshot/restore total mean 1.434 ms, sigma 0.034 ms. It does not use the
older per-layer memcpy loop killed in
[`../errors/2026-05-10-medusa-qwen35-snapshot-ring-step0-killed.md`](../errors/2026-05-10-medusa-qwen35-snapshot-ring-step0-killed.md).

## Params

- Model family: Qwen3.5 CUDA hybrid recurrent state
- Spec verifier rows: `[last_committed_token] + draft_tokens`
- Snapshot source of truth: post-verifier `PagedKVPool::seq_len(slot)`
- Medusa K: pending Phase 1.B core wiring

## Results

This entry is a substrate stub, not a throughput license.

- Snapshot mechanism: CUDA Graph ring replay, prior licensed mean 1.434 ms
- `cargo check --release -p infer --features cuda`: PASS
- `cargo clippy --release -p infer --features cuda -- -D warnings`: PASS
- `cargo check -p infer --no-default-features --features cuda,no-cuda`: PASS
- `qwen35_recurrent_snapshot_ring_restore_idempotent`: PASS

Final greedy-consistency and tok/s license-or-kill benches are deferred to the
full Qwen3.5 Medusa substrate commit, where the draft model path exists.

## License

Not licensed for performance yet. The final Phase 1.B gate remains:

- 0.0% greedy diff across accept lengths `j in {0,1,2,3,4,5}`
- Qwen3.5-4B agent-shape tok/s >= 1.5x vs no-spec for LICENSE
- tok/s < 1.0x vs no-spec for KILL
