# 2026-05-20 — OPD CPU perf: open hand-offs for codex (post-kill update)

> **Audience:** codex (per the 2026-05-20 cooperative split — Claude does
> research / plan / docs / deterministic code; codex does complex code +
> verification). This is the single index of surviving Claude-side
> artefacts from the 2026-05-20 OPD CPU-perf push. **Read this first**
> before opening any of the linked docs.

> **Status as of post-`0a1f945` kill:** the `forward_last_logits` axis
> was killed by codex's production-vocab A/B (0.997 × ± 0.5 % — see
> [`../experience/errors/2026-05-20-forward-last-logits-killed-by-m1-dispatch-hypothesis.md`](../experience/errors/2026-05-20-forward-last-logits-killed-by-m1-dispatch-hypothesis.md)).
> Two open hand-offs remain; one new axis surfaced from the kill.

## Substrate state

The 5-commit chain from `8e8effd` to `15fa6cf` cut per-step matmul from
~30 s to **1.80 s** = 16.7 × cumulative (mixed-dispatch sgemm with
saxpy for `N < 32 768`, `matrixmultiply` for `N ≥ 32 768`;
transpose-aware backward). `lm_head` accounts for ≈ 54 % of the
remaining per-step matmul budget at Qwen3-0.6B.

What the `0a1f945` kill cycle taught us:

- **`lm_head` matmul is not the binding constraint at production
  shape, *at the M values OPD already uses*.** The full rollout
  variant does 3-4 × more FMAs but its wall-clock matches the M = 1
  variant exactly. Either the M = 1 throughput is much worse per FMA
  (likely — see "M-aware dispatch" below), or *some other op* on the
  rollout path is the dominant cost. SOLID requires we don't pick
  between these without a measurement.
- **M-aware dispatch is the new lurking axis.** Existing
  `sgemm_row_major` keys only on N; at N = 151 936 both M = 1 and
  M = 3-4 take the `matrixmultiply` path, but their per-FMA
  throughputs differ. The errors entry above lays out the hypothesis.

## Surviving hand-offs — priority ordered

### P0 — Production `arle train opd` `TensorStore` leak

**Document:** [`docs/research/2026-05-20-opd-production-step-retain-ids-leak.md`](../research/2026-05-20-opd-production-step-retain-ids-leak.md)

Production `run_opd` (and `run_opd_smoke`) call `opd_step` in a loop
with **no `retain_ids` between steps**. Memory grows linearly. At
Qwen3-0.6B a 100-step run leaks ~150 GB before the kernel kicks in.

**Status (2026-05-20 EOD):** codex appears to be implementing exactly
the proposed fix — captured WIP in their tmux session shows
`cleanup_after_backward(store, tape, student_params, &keep_extra)` being
added at the tail of `opd_step` with `keep_extra =
retained_param_and_grad_ids(&teacher_params, store)`. A new
`crates/train/tests/test_opd_step.rs` modification is also in-flight,
plausibly a regression test for the leak.

**Hand-off:** none — codex is finishing this. Claude doc is captured.
Update task list once codex commits.

### P1 — M-aware sgemm dispatch (new — surfaced by the kill)

**Document:** to be written by codex if accepted, with measurement-first
methodology. **Do not start without licensing.**

The kill of `forward_last_logits` produced strong evidence that
`matrixmultiply` at M = 1, N = 151 936 has materially different
per-FMA throughput than at M = 3-4, N = 151 936. Existing dispatch
in `sgemm_row_major` only checks N. A two-dimensional dispatch (M and
N) might:

- Land at saxpy for M = 1 regardless of N (saxpy is good for
  rank-1 vector-matrix), and
- Land at `matrixmultiply` for M ≥ 2, N ≥ 32 768 (the current rule).

**Acceptance criterion (pre-licensed for this axis):** Production-vocab
A/B with mean ≥ 1.05 × on the rollout student forward (matched controls,
same prompt + seed), σ ≤ 2 %. If the M = 1 path benches no faster
than the current matrixmultiply path, kill the axis immediately —
**don't write more code without the M = 1 microbench first**.

**Hand-off:** codex owns the microbench + dispatch rewrite. Suggested
microbench shape: M ∈ {1, 2, 3, 4}, N = 151 936, K = 1 024.

### P2 — `lm_head` transpose-copy hypothesis (still open, lower confidence)

`linear_forward` calls `transpose_host_eager` on the lm_head weight
every invocation, physically allocating + copying a 623 MB row-major
buffer at Qwen3-0.6B. The killed research doc projected this as ~22 %
step saving. **The forward_last_logits kill does NOT refute this
hypothesis** — both A/B variants paid the transpose cost identically;
neither variant tested whether eliminating the transpose helps.

But: the kill also reduces our confidence in any single-variable FLOP /
bandwidth projection on this hot path. Before pursuing this axis,
codex should:

1. **Instrument the transpose call directly** — wrap
   `transpose_host_eager` with an `Instant::now()` timing and counter,
   re-run a single OPD step at production shape, report milliseconds
   spent in transpose vs total step.
2. If ≥ 20 % of step is in the transpose, license the
   `matmul_bt` axis. Derivation already exists in
   [`./2026-05-20-matmul-bt-backward-derivation.md`](2026-05-20-matmul-bt-backward-derivation.md).
3. If < 5 %, kill the axis — the matmul is already cheap and the
   binding constraint is elsewhere.

**Hand-off:** codex owns step 1 (the instrument-and-measure). Steps 2
and 3 are licensing decisions based on step 1's number.

## Killed during this push

- `forward_last_logits` rollout opt — killed by `0a1f945` on 2026-05-20
  per its own SOLID criterion. Lessons captured in errors entry.

## Cooperative protocol notes

From the 2026-05-20 session:

- **OOM under concurrent benches.** Dev box is 31 GB; codex + Claude +
  bench can SIGKILL the smaller process (observed once). **Serialise
  bench scheduling** — peek `tmux` + `free -h` before any large run.
- **Work-split contract.** Claude = research / plan / docs /
  deterministic refactors. Codex = complex code (new autograd ops,
  backward kernels, production-path edits) + verification (bench A/B,
  multi-process runs).
- **Push etiquette.** `git rebase` refuses to operate when working tree
  is dirty, even for unrelated files. When the peer has WIP, either
  wait or coordinate explicitly. **Never `git stash` the peer's
  changes.**
- **License-or-kill pattern (validated this session).** A wins stub
  shipped with an explicit numerical kill criterion → peer measures →
  peer either updates `pending-bench` → `verified` or executes the
  kill. The 2026-05-20 cycle closed cleanly: 7aa11d7 stub → codex A/B
  → 0a1f945 kill. Make this the canonical pattern for "code with
  unverified perf claim."

## Codex resume pointer

Pick up at **P0** (probably already in flight). Once P0 commits, the
production OPD path is leak-free and any subsequent multi-step bench is
trustworthy. **P1** (M-aware dispatch microbench) is the next axis to
investigate — but only the microbench, not the dispatch rewrite, until
the M = 1 throughput number is in. P2 (transpose-copy instrumentation)
can run in parallel with P1's microbench.
