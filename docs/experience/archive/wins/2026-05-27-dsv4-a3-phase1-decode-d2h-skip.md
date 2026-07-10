# DSv4 A3 Phase 1 — decode counts D2H + host scan skip (pending-remote)

## SLO-shape probed? — pending-remote bench

`pending-remote`: This wins entry is a stub. Local probe is not possible
(CUDA build needs nvcc; this Mac does not have one). The pod bench cycle
will run on next available window and the entry will be updated with the
numeric result.

## Context

A3 Phase 1 plan
([`docs/plans/2026-05-26-dsv4-a3-in-graph-metadata.md`](../../plans/2026-05-26-dsv4-a3-in-graph-metadata.md))
target: eliminate `cuMemcpyDtoHAsync_v2` calls per decode token —
current 344-347 (L5 binding constraint), target ≤50.

Prior state: `dsv4_exclusive_scan_i32_cuda` device kernel was already
landed (commit history shows it gated behind `DSV4_A3_PHASE1=1`
default-on), but the surrounding mlp.rs call site continued to D2H the
full `local_counts` vector and do a host-side scan loop **before**
calling the device scan. Net effect: device scan added work; D2H not
removed.

## Change (commit `07305fe9`)

`infer/src/model/deepseek/mlp.rs::DeepseekV4MoeBlock::forward`:

1. Device scan now writes the accumulated total to a 4-byte device slot.
2. Only the 4-byte total is D2H'd (vs the previous full `experts_per_rank
   × 4 B` D2H — at the canonical 64-expert/rank, that's a 64× payload
   reduction *per layer*).
3. `counts_host` / `offsets_host` are built **only** when the downstream
   batched-prefill branch (`forward_compact_local_routes_gpu`, taken at
   `seq_len > 1 && ARLE_DSV4_LOCAL_GROUPED_EXPERTS=1`) actually consumes
   them. The decode hot path (`seq_len == 1`) bypasses the wide D2H +
   host scan entirely.
4. `offsets_gpu` is now produced device-side, so the `clone_htod` from
   `offsets_host` is also skipped on the decode hot path.

## Expected gain (per plan PASS gate)

- NVTX scope at `ffn_route_count_d2h`: previously a wide D2H + host scan;
  now replaced by `ffn_route_total_d2h` (4-byte scalar). Per-NVTX scope
  reduction ≥ 30% expected.
- Per-token decode D2H count: `61 layers × 1 D2H/layer = 61 fewer D2H per
  token` saved on this site (legacy was 1 wide D2H + 1 H2D = 2 transfers
  per layer; A3 is 1 scalar D2H per layer). Toward the ≤50 D2H/token
  binding-constraint target.
- Wall-clock decode tok/s: change-only-here PASS gate is "not worse";
  meaningful improvement waits on Phase 2 (persistent grouped GEMM,
  removes per-expert launch and Site 5 D2H).

## How to validate

`scripts/bench_guidellm.sh dsv4-a3-phase1-decode --model
DeepSeek-V4-Flash` against baseline on 8×H20. Compare:
- nsys NVTX scope `ffn_route_count_d2h` *(removed)* → `ffn_route_total_d2h`
  *(new, expected ≤ 30% of prior wall-clock)*.
- per-token decode wall-clock greedy byte-identical.
- Per-second decode tok/s no regression on c=8 qps=8 32K input/1.5K output.

## Open

- **Phase 2** (persistent grouped GEMM kernel) — replaces 64 per-expert
  GEMM launches with a single launch that reads device counts directly,
  closing the Class A D2H + launch-churn axis. Plan landed but kernel
  not yet wired.
- **Site 5** (recv local count D2H after DeepEP combine) — gated on the
  process-per-rank DeepEP transport landing first
  ([`docs/plans/2026-05-26-dsv4-deepep-process-per-rank.md`](../../plans/2026-05-26-dsv4-deepep-process-per-rank.md)).

## Refs

- Plan: [`../../plans/2026-05-26-dsv4-a3-in-graph-metadata.md`](../../plans/2026-05-26-dsv4-a3-in-graph-metadata.md)
- Project index: [`../../projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md`](../../projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md)
- L5 binding-constraint evidence: [`../../research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md`](../../research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md)
- Commit: `07305fe9`
