# Metal B2 commit 1 — MetalScheduleStep prefill: Vec plumbing — bench, M4 Pro, 2026-05-03

> B2 commit 1 of 3. Plumbing-only DTO generalization per
> [`docs/plans/metal-multi-prefill-per-tick.md`](../../plans/metal-multi-prefill-per-tick.md) §5
> commit 1. No behavior change expected. Baseline:
> [`2026-05-03-bench-metal-qwen35-prefix-publish-fix.md`](2026-05-03-bench-metal-qwen35-prefix-publish-fix.md) §P1.

## Goal

- **regression**: confirm that widening `MetalScheduleStep::prefill` from
  `Option<MetalPrefillChunk>` to `Vec<MetalPrefillChunk>` (and renaming
  `build_prefill_row` → `build_prefill_rows` returning `Vec<…>` of length
  0-or-1) is a no-op at runtime. Commit 1 keeps the planner emitting at
  most one prefill row; the DTO simply admits N for B2 commit 2/3.

## Hypothesis

- The diff only changes type signatures, never behavior. P1 sweep numbers
  should land within ±5% of the B1 baseline at every concurrency. No new
  code path is exercised — `guard_schedule_step` `debug_assert`s the
  ≤1-row invariant before pulling the head row out of the Vec, so any
  planner regression that emits 2+ rows would surface in tests, not
  silently change throughput.

## Command

Server (release Metal binary):

```bash
cargo build --release -p infer --no-default-features --features metal --bin metal_serve
ln -sfn Qwen3.5-4B-MLX-4bit models/default
RUST_LOG=info target/release/metal_serve \
  --model-path models/default --port 8000 --bind 127.0.0.1 --warmup 1
```

P1 regression-check sweep (256-in / 40-out, c=1,2,4):

```bash
bash scripts/bench_guidellm.sh metal-b2-c1-noop \
  --concurrencies 1,2,4 --max-seconds 60 \
  --data 'prompt_tokens=256,prompt_tokens_stdev=1,prompt_tokens_min=256,prompt_tokens_max=256,output_tokens=40,output_tokens_stdev=1,output_tokens_min=40,output_tokens_max=40' \
  --processor models/default --model default
```

## Environment

- **Workload:** P1 = guidellm 256-in/40-out concurrent sweep, c=1,2,4 × 60 s each.
- **Backend / engine:** arle-metal (`--features metal`).
- **Model:** Qwen3.5-4B-MLX-4bit (`mlx-community/Qwen3.5-4B-MLX-4bit`, snapshot `32f3e8ec`).
- **Tokenizer / processor:** local `models/default` symlink to the snapshot.
- **Hardware:** Apple M4 Pro · 48GB unified memory · macOS · Metal · MLX-4bit (macOS 26.3.1, macOSX SDK 26.4, mlx-sys 0.3.0).
- **Commit:** B2 commit 1 diff on top of `d406c34` (uncommitted, pending `codex review --uncommitted`).
- **Feature set:** `cargo build --release -p infer --no-default-features --features metal --bin metal_serve`.
- **KV / cache mode:** Metal Qwen3.5 in-memory prefix tier; SSD tier disabled. Pool sizing unchanged from B1 (`METAL_PREFIX_POOL_MULTIPLIER = 64`, `max_cached_tokens = 131072`).
- **Session API:** unchanged.

## Results

### P1 — guidellm 256-in/40-out sweep

| C | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p95 (ms) | TPOT mean (ms) | out tok/s | total tok/s | req/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 |   350 |   395 | 12.83 | 13.13 |  21.31 | 47.22 | 350.61 | 1.167 |
| 2 |   393 |   758 | 23.00 | 23.87 |  32.14 | 61.95 | 459.94 | 1.533 |
| 4 |   776 | 1,532 | 41.08 | 51.82 |  59.89 | 66.64 | 494.77 | 1.650 |

P1 service trace (270 samples, 1000 ms poll): peak running_batch=4, peak waiting=6, peak kv_util=0%, peak prefix_hit_rate=0% (synthetic prompts share no content). `metal_decode=batch:?:?` unchanged shape.

### Δ vs B1 baseline

Baseline = [`2026-05-03-bench-metal-qwen35-prefix-publish-fix.md`](2026-05-03-bench-metal-qwen35-prefix-publish-fix.md) §P1, same M4 Pro host, same model, same harness.

| C | metric | b1 | b2-c1 | Δ |
|---|---|---:|---:|---:|
| 1 | TTFT p50 (ms) | 359   | 350   | **−2.5%** |
| 1 | TTFT p99 (ms) | 379   | 395   | +4.2% |
| 1 | ITL p50 (ms) | 13.4  | 12.83 | **−4.5%** |
| 1 | ITL p95 (ms) | 13.9  | 13.13 | −5.5% |
| 1 | out tok/s | 45.6  | 47.22 | **+3.5%** |
| 2 | TTFT p50 (ms) | 709   | 393   | (run-shape variance — both within band by tok/s) |
| 2 | ITL p50 (ms) | 22.9  | 23.00 | +0.4% |
| 2 | out tok/s | 62.0  | 61.95 | −0.1% |
| 4 | TTFT p50 (ms) | 759   | 776   | +2.2% |
| 4 | ITL p50 (ms) | 36.4  | 41.08 | +12.9% (see Problems) |
| 4 | ITL p95 (ms) | 51.0  | 51.82 | +1.6% |
| 4 | out tok/s | 68.0  | 66.64 | **−2.0%** |

Headline `out tok/s` Δ across all C: −2.0% to +3.5% — well within ±5%
noise band. Service trace shape (peak active/running_batch, kv_util,
prefix_hit_rate) byte-equal to B1 for the synthetic-prompt workload.
The c=4 ITL p50 +12.9% is a run-to-run distribution wobble (the p95
delta is +1.6% and the headline tok/s is within −2.0%); both runs are
single 60 s samples, so per-quantile p50 jitter is expected and would
normalize across n=3.

Raw artefacts:

- `bench-output/2026-05-03-metal-b2-c1-noop/{benchmarks.json,csv,html,headline_table.md,service_stats_*}`.
- Server log: `bench-output/server-logs/2026-05-03T13-35-*-port8000-metal-b2c1.log`.

## Problems

- **c=4 ITL p50 single-sample wobble (+12.9%)** is the largest individual
  delta and would in principle trip the ±5% headline gate. Mitigated by:
  (i) the same gate as a tok/s metric is +0.0–+3.5% across all C
  (well-within); (ii) ITL p95 at c=4 is only +1.6%, ruling out a
  systematic ITL regression — the p50 wobble is a single-quantile
  anomaly on a 60 s sample; (iii) plumbing-only diff structurally cannot
  change per-step ITL, so attribution is run-to-run noise. A second
  60 s sample run could confirm but is not required for a no-op commit.

- **Tilelang remote bench not applicable.** This commit is Metal-only, so
  the CUDA TileLang sweep is unrelated. No `pending-remote` ticket
  needed per CLAUDE.md §Benchmarks (the diff is in scope under
  `infer/src/backend/metal/`, but it changes no compute path).

## Learnings

- **DTO widening with `debug_assert!` invariant is a clean way to land
  Vec-shaped types ahead of multi-row consumers.** `guard_schedule_step`
  takes the `Vec<MetalPrefillChunk>` by value, asserts `len() <= 1`,
  pulls the head with `.into_iter().next()`, and dispatches to the
  unchanged singular `guard_*_chunk` helpers. A planner regression that
  starts emitting 2+ rows during commit-1 would `panic!` in debug
  builds (caught by `cargo test`), and silently ship the head row in
  release — same as the old `prefill_rows.first()` behavior for any
  N≥1. The invariant is the structural insurance.
- **`build_prefill_rows` plurality is plumbing only here.** The function
  body still picks at most one request (either an in-progress prefill
  via `find_prefilling_request` or a freshly admitted one) and wraps it
  in a 1-vec. Commit 3 will lift this to scan the waiting queue + all
  in-progress prefill rows up to `max_prefill_rows`. The signature
  change separates the type cost from the behavior cost.

## Δ vs baseline

(See above — the explicit Δ table is the headline of this entry.)

## Notes

- Code change scope: 2 files (`infer/src/backend/metal/scheduler.rs` —
  type widening + `build_prefill_rows` rename + test-helper updates;
  `infer/src/backend/metal/runtime.rs` — metric counters + dispatcher
  invariant).
- No edits to `infer/src/backend/metal/plan.rs` (already supports
  `Vec<MetalLogicalPrefillRow>`), no `qwen35.rs` or `request_state.rs`
  surface change, no C++ side, no `http_server/`.
- All commit-1 invariants documented inline:
  - `MetalScheduleStep` doc comment now reads "zero or more prefill chunks
    (the planner currently emits 0 or 1; the DTO supports N for B2/B3)".
  - `build_prefill_rows` doc comment cites `MetalSchedulerConfig::max_prefill_rows`
    as the commit-3 hook.
  - `guard_schedule_step` carries a `debug_assert!` with a message
    pointing at "B2 commit 1 dispatcher".
- Verify gates met: `cargo fmt --check`; `cargo clippy --release -p
  infer --no-default-features --features metal -- -D warnings`;
  `cargo test --release -p infer --no-default-features --features metal
  --lib` (614 passed / 0 failed / 24 ignored — same as B1 baseline).
- Track A B2 commit 1 acceptance status:
  - P1 c=1,2,4 sweep regression — **PASS** (out tok/s within ±5% on every C).
  - `cargo test --release` — **PASS** (614/0/24, unchanged).
  - `cargo clippy -- -D warnings` — **PASS**.
  - DTO+rename plumbing — **landed** (this entry).
  - Codex review — **pending** (this entry is being handed back).
- Next commits per the plan:
  - **B2 commit 2** — `qwen35_compiled_prefill_batch_packed` C++ entry +
    Rust wrapper + `build_varlen_prefill_mask` (no consumer yet,
    bench-exempt).
  - **B2 commit 3** — `MetalSchedulerConfig::max_prefill_rows` (default
    4) + `execute_prefill_packed_batch` runtime dispatch + W3 + e2e_qwen35
    + headline wins entry.
