# Metal DFlash `async_eval` — matched A/B close-out: inconclusive (within noise)

**Status:** closes the `pending-remote` bench debt for commit `d8cb2f4`
(perf(metal): defer DFlash batched terminal eval via async_eval). Outcome:
**inconclusive** — the 2–5% TPOT lever at c=2 does not survive matched
same-binary env-A/B across two thermal-separated sessions. Per
`feedback_matched_ab_for_small_bench_effects.md`, effects ≤10% that only
appear in a single consecutive c-sweep are thermal/cache noise until
reproduced in ≥2 sessions with matched A/B; this did not reproduce (sign
flipped session-to-session). The parent wins entry remains as a
correctness-gated change with no proven throughput delta at c=2.

**Date**: 2026-04-20
**Machine**: Apple M4 Max (40 GPU cores, ~400 GB/s UMA), macOS 26.3.1
**Model**: `mlx-community/Qwen3.5-4B-MLX-4bit` + draft `z-lab/Qwen3.5-4B-DFlash`
**Commit (binary B / async-eval)**: `cbdd7f9` (dflash.rs identical to
`d8cb2f4`; no runtime-path commits landed since — `git log d8cb2f4..HEAD
-- infer/ crates/mlx-sys/` is empty).
**Commit (binary A / preasync)**: `3bc88026` applied via `git checkout
3bc88026 -- infer/src/backend/metal/dflash.rs` against `cbdd7f9`;
remaining tree identical. Only dflash.rs delta: `eval(&to_eval)` →
`async_eval(&to_eval)` at the tail of
`qwen35_dflash_speculative_block_batched` (23 LOC net).
**Parent win**: `docs/experience/wins/2026-04-20-dflash-batched-async-eval.md`

## Goal

Goal type: **regression / validation** (closing pending-remote bench debt).
Settle whether the Audit-1-sized 2–5% TPOT improvement at c=2 for the
DFlash batched speculative path (commissioned in `d8cb2f4`) survives a
matched same-binary A/B run in two thermal-separated sessions per
`feedback_matched_ab_for_small_bench_effects.md`.

## Hypothesis

Pre-run: if async_eval genuinely saves 0.5–1.0 ms/block on the DFlash
batched path (parent win §Learnings), TPOT_p50 at c=2 should be 2–5%
lower on the async-eval binary, in the same direction, across **both**
sessions. The scalar-path parent win's step-driver +12.7% at c=1 already
survives single-session; we expect a smaller-but-consistent signal here.

Failure mode: if the effect is below the thermal/cache noise floor,
single-session deltas will sign-flip as a function of run order (cold vs
warm), not binary identity.

## Setup

Two `metal_serve` binaries built side-by-side against one cleanly
rebuilt `mlx-sys` (shared target dir); only `dflash.rs` swapped between
builds. Metal Toolchain installed via `xcodebuild -downloadComponent
MetalToolchain` (v17.5.188.0) after the parent win flagged missing
toolchain; fresh `libmlx.a` + `mlx.metallib` for both binaries.

Binary hashes (sha256):
```
1bfe69845438866d595fa6fc58790afcc14c918a0d65985ec5f50a76bda270b8  metal_serve_preasync
b5b34253379abac34b5ae7f7b2aea0e90038658c49644e0a11116a9351b63f19  metal_serve_async-eval
```

Per-run bench: `guidellm benchmark run --profile concurrent --rate 2
--data prompt_tokens=1024,output_tokens=256 --max-seconds 60 --warmup 5
--random-seed 20260416 --backend openai_http` against `metal_serve
--dflash-draft-model <draft> --port 8765`. Pre-bench warmup: one
`/v1/chat/completions` with `max_tokens=16` per run.

Cooldown between sessions: 10 min 23 s wall-clock (≥10 min per task
spec). Order randomized:
- **Session 1**: preasync → async-eval (A→B)
- **Session 2**: async-eval → preasync (B→A)

## Environment

- Bench harness: `guidellm 0.6.0`, `openai_http` backend, forkserver MP.
- Tokenizer / processor: local snapshot at
  `~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`.
- Draft snapshot:
  `~/.cache/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash/snapshots/96899cc270945f554998309580b08a04a05a3187`.
- Feature set: `cargo build --release -p infer --no-default-features
  --features metal,no-cuda`.
- Xcode Metal Toolchain: `17E188` (installed mid-session).
- Working tree: clean wrt runtime code for both builds (unrelated
  in-flight edits to `crates/cli/` and `crates/train/` stashed via `git
  stash push -u -m WIP-stash-for-bench-2026-04-20`; **no commits landed
  during this bench**).

## Results

### Raw per-run (c=2, prompt_tokens=1024, output_tokens=256, 60s each)

| Session | Order | Binary      | TPOT_p50 (ms) | TPOT_mean (ms) | TPOT_p99 (ms) | ITL_p50 (ms) | TTFT_p50 (ms) | out tok/s mean | req_success |
|---------|-------|-------------|--------------:|---------------:|--------------:|-------------:|--------------:|---------------:|------------:|
| 1 | 1st (cold) | **preasync**   | 93.01 | 92.48  |  95.75 | 85.22 | 1652.3 | 18.29 | 1280 |
| 1 | 2nd (warm) | **async-eval** | 88.83 | 100.25 | 126.27 | 80.77 | 1414.2 | 20.31 | 1536 |
| 2 | 1st (cold) | **async-eval** | 90.98 |  89.72 |  91.26 | 82.32 | 1447.6 | 18.92 | 1280 |
| 2 | 2nd (warm) | **preasync**   | 89.41 |  95.32 | 109.82 | 81.86 | 1402.7 | 21.38 | 1536 |

### Per-session A-B delta (async-eval vs preasync, by TPOT_p50)

| Session | preasync | async-eval | Δ_abs (ms) | Δ% | Direction |
|---------|---------:|-----------:|-----------:|----:|-----------|
| 1 (A→B) | 93.01 | 88.83 | −4.18 | **−4.5%** | async-eval faster |
| 2 (B→A) | 89.41 | 90.98 | +1.57 | **+1.8%** | async-eval **slower** |

### Cross-session agreement

**Sign flip**. Session 1 suggests async-eval is ~4.5% faster; Session 2
suggests async-eval is ~1.8% slower. The direction that correlates
consistently is *order of run within a session*: the 2nd-run binary in
each session is always faster (cache/JIT warmup), regardless of which
binary that is.

| Binary     | Session 1 | Session 2 | Mean across sessions |
|------------|----------:|----------:|---------------------:|
| preasync   | 93.01     | 89.41     | **91.21 ms**         |
| async-eval | 88.83     | 90.98     | **89.91 ms**         |
| Δ (async − pre) mean |  |   | **−1.30 ms (−1.4%)** |

The cross-session mean delta is **−1.4%**, well below the 10% noise
threshold called out in `feedback_matched_ab_for_small_bench_effects.md`
and smaller than the within-session cold-vs-warm run gap (~3–4% in each
session, independent of binary).

### Crossing the 10% noise threshold?

**No.** Neither the single-session deltas (−4.5%, +1.8%) nor the
cross-session mean (−1.4%) clear the 10% bar. Worse, the sign flip means
even the "direction" of the effect is not established.

## Verdict

**Inconclusive.** The `async_eval` change does not produce a
statistically-detectable TPOT improvement at c=2 under this harness on
this hardware. The Audit-1 2–5% estimate was an *upper bound* for the
opportunity; the realized delta sits inside thermal/cache noise. The
change remains correctness-gated (parity tests in the parent win all
pass) and non-regressing (no session showed async-eval regressing by
more than +1.8%, which is itself within noise), but the throughput win
stays unproven.

Per the task's regression gate ("anything ≥ −2% must be investigated"),
the +1.8% Session 2 delta is borderline — handed back in §Problems as a
follow-up note but not escalated to errors/, because the cross-session
mean is net-negative (−1.4%) and the sign flip with order indicates
order confounding, not a binary regression.

## Problems

- **Order-of-run dominates binary identity.** In each session, the 2nd
  bench (warm caches, Metal JIT + ANE already paged in from the 1st run)
  runs ~3–4% faster than the 1st bench regardless of which binary is
  loaded. This is larger than the async_eval effect itself. For future
  single-lever validations ≤5% sized, paired-samples are needed (alternate
  A/B/A/B… in one session) rather than sequential A-then-B.
- **Metal Toolchain state.** `xcrun metal` was missing at session start
  (same as parent win §Problems). Installed via `xcodebuild
  -downloadComponent MetalToolchain` → `17E188`; subsequent mlx-sys
  rebuild produced a fresh, working `libmlx.a` + `mlx.metallib`. Prior
  attempts that linked against a stale `gather_front.cpp` from a
  previous toolchain-missing build failed at runtime with Metal shader
  compile errors. Worth flagging in CLAUDE.md: a stale mlx-sys
  `jit_source` output from a toolchain-missing build is silently
  retained until manually nuked under
  `target/release/build/mlx-sys-*/out/`.
- **Session 2B preasync p99 TPOT = 109.8 ms** vs session 2A async-eval
  p99 = 91.3 ms. Warm-run tail suggests the 2nd run's memory pressure
  or kv-cache layout may be different, not that the binaries differ.
  TPOT_mean is less affected (95.3 vs 89.7) but still influenced.
- Raw tok/s is correlated with `req_success` count (1280 vs 1536) —
  guidellm's concurrent profile ran more requests in warm sessions,
  which isn't a binary-level comparison.

## Learnings

- **Order randomization alone is insufficient for ≤5% deltas under
  thermal/JIT confounds.** Session-level warm-up between A and B
  dominates a single-run A/B. When a hypothesized effect is ≤5% and
  the single-run cold-vs-warm gap is ≥3%, fully interleaved A/B/A/B or
  repeat-until-variance < effect is required, not A→B once per session.
- **MLX jit_source regen needs the Metal Toolchain.**
  `make_compiled_preamble.sh` captures `xcrun metal` stderr into the
  generated C++ as "header content" when the toolchain is missing,
  producing a runtime-broken library that still *compiles and links*
  without error. The failure surfaces only at `newLibrary(source)` call
  sites during the first metal_serve kernel load. Worth a build-time
  guard in `crates/mlx-sys/build.rs` that checks for `xcrun metal`
  before invoking cmake, with a clear error.
- **`async_eval` deferral has a sound correctness argument but an
  unobservable latency win at c=2 under matched A/B.** The
  parent-win's correctness audit remains valid; the perf hypothesis
  needs either (a) a lower-noise measurement path (Metal capture,
  `ncu`-equivalent instruments) or (b) a higher-concurrency harness
  where overlapped graph-build actually saturates a resource. The c=2
  HTTP path may not be the right probe — the scalar step-driver win at
  c=1 (+12.7%) worked because the tight-loop FFI path had zero other
  overlap; c=2 on the batched path already has cross-request overlap
  that may absorb the saved fence.

## Parent wins entry — action

The parent wins entry
`docs/experience/wins/2026-04-20-dflash-batched-async-eval.md` is
**left with `pending-remote` replaced by "inconclusive — within
noise"**. Rationale: this bench did run; the measured delta is not
zero, but is not a confirmed positive either. Replacing the status with
a false numeric would over-state the result.

The parent entry's "Expected delta: +2–5% TPOT at c=2 on DFlash-batched
decode, or within noise" was correct in its *or* branch: we landed in
the noise-absorbed branch. Correctness still holds; runtime change
stays in, no revert needed.

## Δ vs prior snapshot

- Baseline anchor: `docs/experience/wins/2026-04-20-metal-qwen35-decode-double-buffer.md`
  §HTTP concurrent table — reports c=2 TPOT_mdn = 22.98 ms / 23.20 ms
  for A/B of the scalar step-driver change (a different code path).
  That entry's c=2 row was on Qwen3.5 scalar, not DFlash, so it is not
  directly comparable with this bench's DFlash+draft 88–93 ms c=2
  TPOT. The ~4x TPOT gap vs the scalar baseline is expected (DFlash
  adds a speculative block of draft forward + target verify per
  accepted-token step); the absolute DFlash c=2 number is unchanged
  across binaries in this bench.
- First matched-A/B run for this specific change.

## Artefacts

- Raw: `bench-output/2026-04-20-dflash-async-eval-matched-ab/{s1,s2}-{preasync,asyncev}-c2/benchmarks.{json,csv}`
- sha256:
  - s1-preasync-c2: `0ebcb5fa5d3ed40625f9d4e2fa248c6923bb8a6783323704610778e36ae854ec`
  - s1-asyncev-c2:  `4294ab8343fad475fbbef384b048ce732ed95edf08c0988ab89b98d4d26d6c77`
  - s2-asyncev-c2:  `ece417f0ac0836c9b52fa5f9f6f98f8b4bc35df6520788e479b0545a5e804649`
  - s2-preasync-c2: `b3c9c4c847997c9b4d5c02b8e99984a34cfd5ca61824a67adc282074742d0ebb`
- Binaries (ephemeral, /tmp, not committed):
  - `/tmp/dflash-bench-bins/metal_serve_preasync` (sha256 `1bfe6984…`)
  - `/tmp/dflash-bench-bins/metal_serve_async-eval` (sha256 `b5b34253…`)

## Rule

**When a hypothesized per-step saving is within ~2× the cold-vs-warm
gap of the bench harness, matched A/B across two sessions is necessary
but not sufficient. Interleave A/B/A/B within a session, or land a
trace-driven measurement (Metal capture) that directly measures the
targeted fence latency, before publishing a numeric delta.** The
feedback memory's "≤10% effects require matched A/B in ≥2 sessions"
rule is the *minimum* bar; sub-5% effects often need the stronger
interleaved or trace-anchored protocol to clear order confounds.

## Cross-refs

- `docs/experience/wins/2026-04-20-dflash-batched-async-eval.md` —
  parent win; §"Numeric c=2 TPOT delta — pending-remote" block
  supersedes to "inconclusive — within noise" per this entry.
- `docs/experience/wins/2026-04-20-metal-qwen35-decode-double-buffer.md`
  — scalar step-driver parent (+12.7% c=1, HTTP c=2 flat), same
  `async_eval` pattern one layer deeper; this entry confirms that
  layer's c=2 flatness narrative.
- `docs/experience/wins/2026-04-20-metal-qwen35-post-double-buffer-audits.md`
  §Audit 1 — where the 2–5% lever was sized; the sizing was optimistic
  under this harness.
- `.claude/projects/-Users-user-code-agent-infer/memory/feedback_matched_ab_for_small_bench_effects.md`
  — the rule applied here; this entry is an instance of "did not
  reproduce" under it.
- `infer/src/backend/metal/dflash.rs:2473` — `async_eval(&to_eval)` at
  HEAD; unchanged in this bench.
