# Runtime contract unification (HTTP + CLI on RequestHandle) — guidellm sweep, cuda, 2026-04-27

> Status: **pending-remote** — refactor collapses CUDA from a deprecated
> single-request `ModelInferenceEngine` path onto the same `RequestHandle`
> + `IncomingRequest` contract Metal/CPU already use. The actual HTTP
> serve path was already on `SchedulerHandle` directly and is **unchanged**
> by this batch; the unified `LoadedInferenceEngine::Cuda` variant only
> changes the **CLI agent** + **E2E test** loaders. Local Mac runs Metal,
> can't exercise the CUDA scheduler hot path.

## Goal

- Regression-check that the runtime-contract unification has **zero**
  measurable impact on serving throughput. Goal type: regression-check
  minimum (per `bench-and-trace-spec.md` §7).
- Cover commit range `9e8ef9a..26ad9db` (4 refactor commits, with one
  unrelated user commit `72806dc perf(metal): tune gguf qwen35 decode`
  interleaved — that one is Metal-only and won't show on a CUDA sweep).

## Hypothesis

All four commits are pure refactors with no algorithmic delta on the HTTP
hot path:

- **9e8ef9a** `refactor(http): thread session_id through CompletionRequest,
  drop hardcoded None` — adds `pub session_id: Option<SessionId>` and
  `pub trace_context: Option<SpanContext>` fields to `CompletionRequest`
  (default `None` on every existing caller). Removes the
  `RequestHandleInferenceEngine::submit_request` hardcoded `None`
  fall-through. **HTTP path unchanged** because HTTP constructs
  `IncomingRequest` directly via
  `RequestExecutionOptions::into_incoming_request`, which has always
  threaded `session_id` correctly.

- **64205c7** `refactor(scheduler): add unified CUDA Cuda(RequestHandle)
  variant on LoadedInferenceEngine` — adds a new
  `LoadedInferenceEngine::Cuda { engine: RequestHandleInferenceEngine<SchedulerHandle>, _guard: SchedulerRuntimeGuard }`
  variant alongside the legacy `Qwen3 / Qwen35 / Qwen35Moe`. **HTTP serve
  path doesn't use `LoadedInferenceEngine`** (`infer/src/main.rs` calls
  `spawn_scheduler_handle_from_path` directly). The new variant is a
  CLI-agent + test surface only.

- **fab93b6** `refactor(tests): migrate E2E to LoadedInferenceEngine
  scheduler path` — 4 production E2E + 2 data-gen tests now load via the
  scheduler. **No production code touched.**

- **26ad9db** `refactor(server-engine): delete ModelInferenceEngine and
  per-model aliases` — deletes 871 LOC of deprecated single-request code
  (struct, impls, type aliases, three legacy `LoadedInferenceEngine`
  variants, `vocab_size()` test instrumentation, `prepare_with_prefix_cache`
  + `cached_prompt` private prefix-reuse logic). **No production code
  path lost** — RadixCache in the scheduler covers prefix reuse.

  Expected Δ on canonical Qwen3-4B sweep: **0% TTFT / 0% ITL / 0%
  out-tok-s within run-to-run noise (≤1.5%)**.

## Local Metal sanity-check (regression-only, 2026-04-27)

- `cargo check --workspace` — clean across all four feature combos
  (`metal`, `cuda,no-cuda`, `no-cuda`, `no-cuda,cpu`).
- `cargo clippy --workspace -- -D warnings` — clean.
- `cargo test --release --lib -p infer` — 423 passed, 0 failed
  (down from 425 because two tests targeting now-deleted private helpers
  — `test_truncate_at_stop`, `choose_prefix_reuse_action_covers_scheduler_aligned_cases`
  — were removed alongside the deletions).

Apple Silicon doesn't exercise the CUDA scheduler runtime, so Metal lane
proves only "doesn't break Metal compile + lib tests", not "CUDA hot path
unchanged". The full Δ verdict requires the remote bench below.

## Command

```bash
scripts/bench_guidellm.sh cuda-runtime-contract-unify
```

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** TBD (whichever remote CUDA box runs the sweep)
- **Commit:** `26ad9db` (HEAD of the four-commit refactor batch
  `9e8ef9a..26ad9db`)
- **Feature set:** `cargo build --release` (default cuda)
- **Non-default flags / env vars:** none
- **Server launch:** `scripts/start_infer.sh Qwen/Qwen3-4B 8000`

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`

## Results — pending

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| pending-remote | — | — | — | — | — | — |

## Δ vs baseline — pending

Baseline: most recent CUDA Qwen3-4B guidellm sweep before this batch
landed (whichever ship snapshot is newest under
`docs/experience/wins/*-bench-guidellm-cuda-*.md` referencing the same
canonical params). Expect ≤1.5% Δ on every column (regression-check bar).

## Problems / Learnings

- The unification narrative is "HTTP and CLI now share one runtime
  contract". The reality after Phase 3 is more nuanced: HTTP was
  **already** on `SchedulerHandle` (and never went through
  `LoadedInferenceEngine`). What this batch actually unifies is the
  **CLI-agent + E2E test surface** — those previously called the
  deprecated `ModelInferenceEngine::complete` and now go through the same
  `RequestHandle` adapter Metal uses.
- Net deletion: −817 LOC across 4 commits (3a +39, 3b +36, 3c −21,
  3d −871). The cleanup mostly comes from removing the parallel
  single-request path (`ModelInferenceEngine`, `Qwen3InferenceEngine`,
  `Qwen35InferenceEngine`, three `LoadedInferenceEngine` variants, and
  the `prepare_with_prefix_cache` / `cached_prompt` helpers).
- If any column moves >1.5%, bisect across 9e8ef9a..26ad9db; the most
  plausible suspect would be 64205c7 (introduces the
  `SchedulerRuntimeGuard` thread spawn on `LoadedInferenceEngine::load`)
  if any test or downstream caller hits that path on a hot loop.

## Rule

- Per `feedback_bench_every_change.md`: in-scope diffs (anything under
  `infer/src/`) require a wins/ entry. Refactors that touch
  `server_engine.rs` and `scheduler/cuda/` are firmly in scope even when
  the production HTTP path is not on the affected variant — defending
  the no-regression bar protects against silent ABI / call-shape
  surprises from the deletion side.
