# prefill snapshot cuts and row slicing land in infer-plan

Date: 2026-09-10 · Lane: lane/d10 · Step 3a slice 6 (blocks 1+2 → `infer-plan`)

## Context

`prefill_row_snapshotted` chunks a hybrid prefill at recurrent-state
snapshot boundaries so a later prefix hit can restore the linear-attn
state. The cut positions — stride multiples inside the chunk, the
start-aligned boundary, and `L*` (the last page-aligned position before
`total_tokens`) on the final chunk — were computed inline, as was the
`PrefillRow` segment/tail construction (two hand-built struct literals
slicing `tokens` by absolute position).

## What worked

- **`prefill_snapshot_cuts(start, end, total_tokens, page_size,
  stride_pages) -> Vec<(usize, bool)>` is the cut plan.** The stride
  arithmetic, the `L*` tag, and the sort are pure host over token counts;
  the caller keeps the device half — materialize recurrent state at each
  cut, store it by the `is_lstar` tag, and forward the segments. The
  function is backend-neutral: the executor passes its page size and
  stride, infer-plan stays device-free.
- **`PrefillRow::slice(from, to)` names the segment construction.** The
  two literals (segment and tail) differed only in the token range; the
  method takes absolute positions and hides the `from - start_pos`
  index arithmetic.

## Rule

Chunked-prefill cut positions are host arithmetic over token counts:
return them as a sorted `(position, tag)` plan and let the caller run the
state materialization. A struct literal that slices `tokens` by absolute
position and clones the rest is a `slice(from, to)` method — the index
arithmetic is the only non-obvious part, and it belongs on the type.

## Net

No baseline: refactor, no measurement; test counts and diff stat are
mechanical facts.

`git diff --stat` vs the slice-5 commit: +107/−43 = +64 net across 2
files (infer-plan +98 with 2 new tests, qwen35.rs −34 — the cut
computation and both row literals shrink). `cargo test -p infer-plan`
16/16 green; Mac CUDA clippy gate exit 0; `check_repo_hygiene.py` green;
`cargo fmt --check` clean.
