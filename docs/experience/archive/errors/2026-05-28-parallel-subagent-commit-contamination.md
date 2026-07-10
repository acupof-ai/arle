# Parallel subagents sharing a working tree co-mingled commits — GAP-A plan doc commit `ab850f7a` carries GAP-C's FP8 kernel change

## SLO-shape probed? — N (process/hygiene incident; no perf signal)

## TL;DR

Three GAP subagents (A, B, C from
[`docs/research/2026-05-28-arle-kernel-vs-sota-audit.md`](../research/2026-05-28-arle-kernel-vs-sota-audit.md))
were dispatched in parallel as background `general-purpose` Agent calls.
All three shared the same on-disk working tree — the Agent tool defaults
to a shared workspace; `isolation: "worktree"` is opt-in. Despite each
brief mandating explicit-path `git add`, the GAP-A subagent's
"Phase 1 plan doc" commit absorbed GAP-C's already-edited-on-disk FP8
kernel change. Result:

- **`ab850f7a`** message says `docs(cuda): plan GAP-A CUTLASS-MMA quantized GEMV port`
  but the diff contains 111 LoC of FP8 cp.async kernel work in
  `crates/cuda-kernels/csrc/attention/decode_attention_quantized.cu`
  plus the GAP-A 229-line plan doc.
- **`489cc583`** (follow-up by GAP-C subagent) is the FP8 cp.async
  wins entry only; its message cross-references `ab850f7a` as the
  actual code carrier, but `git blame` on the FP8 lines will still
  resolve to `ab850f7a` with the misleading message.

The history is already on `origin/main`; per CLAUDE.md no-amend-after-push,
neither commit will be rewritten. This entry is the durable cross-ref.

## What `ab850f7a` actually contains

```
docs(cuda): plan GAP-A CUTLASS-MMA quantized GEMV port

 crates/cuda-kernels/csrc/attention/decode_attention_quantized.cu | 111 ++++++----   ← GAP-C-cheap, NOT GAP-A
 docs/plans/2026-05-28-gap-a-cutlass-mma-quant-gemv.md            | 229 +++++++++++   ← actual GAP-A plan doc
 2 files changed, 301 insertions(+), 39 deletions(-)
```

The 111-LoC kernel change implements the FP8 cp.async pipeline
mirroring the INT8 sibling (`__pipeline_memcpy_async`-driven
double-buffered K/V tile prefetch on `decode_attention_fp8_per_channel_k_partial_kernel`).
That is GAP-C-cheap per the audit, not GAP-A. The commit message is
silent on this content.

## Why it happened — parallel subagents share the working tree

The Agent tool launches subagents in the same cwd by default; only
`isolation: "worktree"` materializes a separate git worktree. When
multiple `general-purpose` subagents are running impl work in parallel:

1. Subagent C edits `attention/decode_attention_quantized.cu` (in
   memory / on disk via the Edit tool). At this point its edits are
   **visible to all other subagents and to the shared `git status`**.
2. Subagent A — concurrently — stages its own GAP-A files. Even with
   the brief's "explicit-path `git add`" rule, A staged its own paths
   AND the plan doc; the FP8 file was already touched on disk and
   ended up in the unified index commit if A then ran
   `git commit -m ...` without inspecting `git diff --cached` for
   contamination.

Both subagents reported afterwards. C noticed the race in its post-flight
report; A's report (still pending at time of writing) had already
pushed the contaminated commit.

## Fix recipe — going forward

**Hard rule** (durable, captured in `~/.claude/.../feedback_parallel_subagents_share_worktree.md`):

> Multi-subagent impl-parallel dispatch MUST use `isolation: "worktree"`
> on each Agent call. The shared-tree default is fine only for:
> - read-only research (Explore)
> - serial dispatch (one subagent at a time)
> - mixed dispatch where only one of the running subagents is impl

**Recovery on history**: do not amend, do not revert (`ab850f7a` carries
the actual FP8 cp.async kernel — reverting would also rip out the
working code). Cross-ref via wins/errors entries is the only safe
remediation post-push.

**Detection** (lesson): a subagent that runs
`git diff --cached` immediately before `git commit` and pattern-matches
the staged files against its expected file list will catch
contamination in time to abort the commit. Add to all future impl
subagent briefs.

## What `git blame` will show on the FP8 lines

Anyone running `git blame crates/cuda-kernels/csrc/attention/decode_attention_quantized.cu`
on the FP8 cp.async lines (`__pipeline_memcpy_async`, `smem_k[2][TILE_TOKENS][HEAD_DIM]`,
`preload_page`, etc.) will see `ab850f7a docs(cuda): plan GAP-A ...`.
**That is the right author and the right commit for the code, but the
message is misleading.** The actual content provenance is the GAP-C-cheap
work documented in
[`docs/experience/wins/2026-05-28-cuda-gap-c-cheap-fp8-cpasync.md`](../wins/2026-05-28-cuda-gap-c-cheap-fp8-cpasync.md).

## Rule

When dispatching multiple parallel `general-purpose` impl subagents:
1. Set `isolation: "worktree"` on each Agent call — non-negotiable.
2. Each subagent's brief must include `git diff --cached` self-audit
   immediately before `git commit`, with a hard requirement that the
   staged paths match the subagent's declared file list.
3. If a subagent reports race-contamination in flight: do not attempt
   to amend or revert post-push; write a cross-ref entry making the
   misleading commit's actual content discoverable.

## Refs

- Commit timeline:
  - `bc068c93` (20:?? GAP-B clean — single-subagent landing, no race)
  - `ab850f7a` (20:40:16 GAP-A plan-doc message, GAP-C kernel content)
  - `489cc583` (20:45:02 GAP-C wins entry — cross-references `ab850f7a`)
- Subagent reports: GAP-C (`aa0f7d2c17eba1385`) explicitly flagged the
  contamination on completion; GAP-A still in flight at time of writing.
- Tool behavior: Agent tool `isolation` parameter — set to `"worktree"`
  to materialize a separate git worktree per subagent.
