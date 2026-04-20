# Metal DFlash @ HEAD — baseline vs on at c=1

**Date**: 2026-04-20
**Machine**: Apple M4 Max (40 GPU cores, ~400 GB/s UMA), macOS 26.3.1
**Binary**: commit `9978182` (HEAD), `metal_bench` built incrementally from
cached `libmlx.a` (fresh Metal Toolchain absent — pre-existing env state,
see `2026-04-20-dflash-batched-async-eval.md` §Problems).
**Target**: `mlx-community/Qwen3.5-4B-MLX-4bit`
**Draft**:  `z-lab/Qwen3.5-4B-DFlash`

## Goal

Settle the "DFlash on vs off" c=1 delta at current HEAD as a single-session
directional bench. Replaces the `pending-remote` preasync-vs-HEAD matched-A/B
ticket from `2026-04-20-dflash-batched-async-eval.md` — that attempt failed
because the subagent couldn't fresh-build the preasync commit without the
Metal Toolchain. This bench answers the **practical user question** ("DFlash
on, how much does it help at HEAD today?") instead of the commit-attribution
question ("how much did `async_eval` alone contribute?").

## Hypothesis

With `f6be5f6` (double-buffer) + `d8cb2f4` (async_eval) + `3bc8802`
(prefill-fastforward) all stacked on HEAD, DFlash-on at c=1 should be
materially above DFlash-off. Prior terminal-state docs put step-driver c=1
at ~85 tok/s; HTTP/bench decode usually lands 10–20% below that.

## Params

- `--prompt-tokens 32 --generation-tokens 256 --warmup 1 --runs 3`
- c=1 (metal_bench is single-request by design — c≥2 requires
  `metal_serve` + `guidellm`, out of scope for this pass)
- Build: `cargo build --release -p infer --no-default-features
  --features metal,no-cuda --bin metal_bench`

## Env

- Binary incrementally built at 14:40 local, 8.0 s via cached `libmlx.a`
  (fresh Metal Toolchain `xcrun metal` still errors in this env; same
  state as `2026-04-20-dflash-batched-async-eval.md`).
- No other GPU load during the run (bench-off check: idle GPU before
  launch).
- Single consecutive session, B/A order: baseline first, DFlash second.
  Gap between runs: ~30 s (embedded in binary setup time).
- No matched A/B across sessions — **the +78% effect is well above the
  10% thermal-noise threshold from
  `feedback_matched_ab_for_small_bench_effects.md`, so the single-session
  reading is directionally sufficient**.

## Results

Raw output: `/tmp/bench-dflash-c1/run.log` (kept for this session).

| Metric                 | Baseline (DFlash off) | DFlash on        | Δ        |
|------------------------|-----------------------|------------------|----------|
| Generation TPS (mean)  | 41.3 tok/s            | **73.5 tok/s**   | **+78%** |
| Generation TPS (p50)   | 41.6 tok/s            | 73.2 tok/s       | +76%     |
| Generation TPS (p99)   | 41.7 tok/s            | 74.0 tok/s       | +77%     |
| TTFT (mean)            | 107 ms                | 62 ms            | −42%     |
| Total wall (mean)      | 6309 ms               | 3547 ms          | −44%     |
| Repo E2E TPS (mean)    | 40.6 tok/s            | 72.2 tok/s       | +78%     |
| Peak RSS               | 2346 MB               | 2514 MB          | +7%      |

The baseline 41 tok/s is lower than the 2026-04-20 step-driver c=1 figure
of 85 tok/s because this bench runs the *full* `metal_bench` path
(runtime + sampler + bookkeeping), not the tight step-driver FFI loop
that the double-buffer win measured. The DFlash-on line at 73 tok/s
sits in the right ballpark for block-verify speculative decoding with
the expected ~2× acceptance-weighted decode speedup.

## Problems

- **No c≥2 coverage in this pass.** `metal_bench` is single-request;
  `metal_serve` + `guidellm` is the right rig for c=2..c=8 — ~30–60 min
  per sweep. Deferred.
- **No preasync-commit comparison landed.** Background agent `a007ae4…`
  died attempting the fresh build at `d8cb2f4^`; worktree was pruned.
  `--baseline-compare` flag work-in-progress on another agent
  (`aff9906…`) will make this a single-invocation one-shot going forward.
- Pre-existing `cargo clippy --features metal` cmake profile drift is
  still unresolved — noted in prior wins entries, not introduced here.

## Learnings

- **When the effect is large (≥2×), single-session is enough.** Matched
  A/B across sessions is the right rule for ≤10% effects where thermal
  noise dominates — it's overkill when the answer is "DFlash nearly
  doubles c=1 TPS on this machine, today, at HEAD".
- **Pin the binary before launching a long bench.** The bench agent that
  died was using `cargo run` which would have hot-recompiled had the DX
  agent committed to `infer/` mid-run. This pass built `metal_bench`
  explicitly with `cargo build` first, then invoked the produced binary
  directly. Zero chance of mid-bench drift.
- **`metal_bench --baseline-compare` should land.** DX-polish agent
  `aff9906…` is implementing it. One invocation → two-row output + Δ%
  is strictly better than today's "run twice, copy-paste numbers" UX.

## Cross-refs

- `docs/experience/wins/2026-04-20-dflash-batched-async-eval.md` — the
  `pending-remote` matched-A/B debt this pass partially clears (c=1 only).
- `docs/experience/wins/2026-04-20-metal-qwen35-decode-double-buffer.md` —
  `f6be5f6`, the scalar-path double-buffer win stacked into this number.
- `docs/resources/metal-dflash.md` — user-facing canonical doc; the c=1
  numbers here are consistent with its "Beta — default-on" claim.
- `scripts/run_dflash.sh` — canonical runner (this pass invoked
  `metal_bench` directly to pin the binary; `run_dflash.sh bench` would
  have `cargo run` and risked mid-DX-agent recompilation).

## Rule

**When DFlash is effectively 2× at c=1 in a single-session read, that is
the ship-it number. Cross-session matched-A/B exists for the 2–5% regime.
Reserving it for big effects is noise-chasing.**
