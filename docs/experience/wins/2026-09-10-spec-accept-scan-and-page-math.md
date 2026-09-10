# spec accept scan and page math land as pure values in infer-plan

Date: 2026-09-10 · Lane: lane/d8 · Step 3a slice 4 (blocks 1+2 → `infer-plan`)

## Context

Slice 3 (#271) moved the dispatch ladder and prefill geometry. The two
largest mixed functions left are `dspark_decode_batch` (361 lines) and
`mtp_decode_batch` (278 lines); both end in the same accept loop — scan the
verified argmax for the longest matching draft prefix, emit the accepted
drafts plus the bonus, and record a rollback entry on a partial accept. The
greedy half of that scan lived in `dspark_accept_commit` (qwen35/dspark.rs),
a pure-host function with a vestigial `&self` that the mtp path had inlined
as its own copy. Four call sites also open-coded `len.div_ceil(page_size)`
for the page count a mirror must cover.

## What worked

- **`spec_accept_greedy` returns a `SpecAcceptOutcome` value; the caller
  executes it.** The scan (longest matching prefix over `&[u32]`, the bonus
  from the first miss or the final argmax, the `partial` flag) is pure host
  arithmetic. Both batch functions now call it and keep only the device
  leaves: the next-hidden copy, `truncate_slot`, `mirror_slot`, and the
  rollback batch. `dspark_accept_commit` disappears; the mtp inline copy
  disappears with it.
- **The rollback decision crosses as data, not as a branch.** `partial`
  plus `k` is everything the caller needs to push `(slot, start, k)` onto
  the rollback vec; the rollback batch itself stays a device call.
- **`pages_covering(len, page_size)` names the page math.** The four
  mirror/error sites that open-coded `div_ceil` now call one tested
  function; the name says what the number is for.
- **The sampled branch stays put.** `dspark_accept_commit_sampled` reads
  logits and runs a rejection kernel — it is a device call, not host
  arithmetic, so only the greedy twin moved.

## Rule

The accept loop of a speculative-decode batch is host arithmetic over two
`&[u32]` slices; extract it as a pure function returning the emitted
tokens, the bonus, and the partial-accept verdict, and leave the caller
the device leaves. A pure-host function with a vestigial `&self` is a
extraction candidate by definition — the receiver is the tell.

## Net

No baseline: refactor, no measurement; test counts and diff stat are
mechanical facts.

`git diff --stat` vs the slice-3 commit: +120/−55 = +65 net across 5 files
(infer-plan +98 with 6 new tests, dspark.rs −33 from the deleted
`dspark_accept_commit`, qwen35.rs roughly flat — the accept loops were
rewired, not removed). `cargo test -p infer-plan` 12/12 green; Mac CUDA
clippy gate exit 0; `check_repo_hygiene.py` green; `cargo fmt --check`
clean.
