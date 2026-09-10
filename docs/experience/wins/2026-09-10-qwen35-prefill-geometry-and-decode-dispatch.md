# qwen35 prefill geometry and decode dispatch land as pure values in infer-plan

Date: 2026-09-10 · Lane: lane/d7 · Step 3a slice 3 (blocks 1+2 → `infer-plan`)

## Context

The architecture refactor (docs/plans/2026-09-09-architecture-refactor.md §3.2)
splits the qwen35 executor along one recurring boundary: host arithmetic and a
device call fused in one body. The arithmetic moves to the pure-data
`infer-plan` crate; the device call stays in `infer-cuda` as a thin caller.
Two types the plan names as missing blocked step 3b:

1. `build_prefill_geometry` computed CP-ring slices, page indices, and q/k
   position arrays inline, then uploaded them in the same body.
2. `dispatch_decode_rows` was the `dspark → mtp → plain` ladder as a match
   that executed immediately; the graph-capturable judgment (pool format +
   FA3 state) was a third copy of the same format knowledge, inlined inside
   `try_graph_decode_paged`.

## What worked

- **`PrefillGeometry::compute` is the stage-2 output.** Balanced block-cyclic
  CP slices (`slices[p] = (p·base + p.min(rem), base + (p < rem))`), this
  rank's q positions, per-owner k positions, and the slot page indices — all
  pure host values, no `ctx`, no device handle. `build_prefill_geometry`
  shrinks to the upload half: `PageMeta::for_ring_prefill` / `for_slot` plus
  the `upload_f32` / `upload_i32` calls that build `Qwen35CpPrefill`.
- **`decide_decode` returns a `DecodeDispatch` value; `execute_decode` runs
  it.** The ladder's inputs (spec kind, row count, compatibility, greediness,
  KV class, `--spec-max-batch`) are all host facts, so the route is a pure
  function. The executor calls `decode_dispatch` then `execute_decode` — the
  decision is inspectable and testable without a device.
- **The capturable judgment joins the value via `DecodeKvClass`.** The
  executor maps its pool format + FA3 state onto five classes
  (`NoPool`, `Bf16Fa3`, `Bf16NoFa3`, `Quantized`, `Other`); `decide_decode`
  reads only the class. `PlainSingle { capturable }` carries the verdict to
  `submit_decode_row`, and `try_graph_decode_paged` loses its inline format
  match — the armed flag and the seq-len gate stay with the graph slot,
  where the state they guard lives.
- **The rest of block 1's host arithmetic moves with the ladder.**
  `speculative_chain_fits`, `qwen_spec_decode_compatible`, `SpecChain` +
  `assign_row_offsets` + `flatten_chains`, and `spec_accept_totals` (the
  pending row is never a reject) all become `infer-plan` functions with
  unit tests at the definition site. The two tests that lived in
  qwen35.rs's `tier_io_tests` module move with their subjects; the module
  disappears.

## Rule

When a function fuses host arithmetic with a device call, split at the
boundary: the arithmetic becomes a pure value in the data crate, the call
stays a thin executor method. A dispatch ladder is a value, not a match
branch — return the decision, let the caller execute it, and every format
fact the decision needs crosses the seam as a backend-neutral class, never
as the backend's own enum.

## Net

No baseline: refactor, no measurement; test counts and diff stat are
mechanical facts.

`cargo test -p infer-plan` 10/10 green (4 new tests: geometry slices,
non-ring layout, the dispatch table, accept totals); Mac CUDA clippy gate
exit 0; `check_repo_hygiene.py --selftest` 9/9 green.
