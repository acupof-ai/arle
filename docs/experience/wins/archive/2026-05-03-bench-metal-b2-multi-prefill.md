# Metal B2 commit 3 — multi-prefill exploration: NEGATIVE RESULT + REVERT — M4 Pro, 2026-05-03

> **Acceptance gate MISSED + opt-in code REVERTED.** Multi-prefill was
> wired through scheduler + runtime per the plan, exercised against W3
> with `--metal-max-prefill-rows 4`, and **did not deliver a warm-TTFT
> win** under the "divide budget evenly across N rows" policy (warm p99
> 30.8s vs target 4s, regression cold p50 +22%). After the negative
> bench landed, the patch went through six rounds of `codex review
> --uncommitted` that surfaced 2 P1 + 5 P2 + 1 P3 across the multi-prefill
> opt-in path; each fix opened a new flank (the v6 surplus-DFlash-row
> cancellation regression was introduced by the v5 P2-1 fix itself).
>
> **Decision: revert the c3 multi-prefill machinery; keep this entry as
> the diagnosis record.** The B2 c1 (`Vec` DTO at commit `f82daf3`) and
> B2 c2 (`qwen35_compiled_prefill_batch_packed` C++ entry at commit
> `a896c31`) stay shipped — they are clean reusable infrastructure for a
> future B3 commit. What this commit deletes is the scheduler-side
> admission logic, the runtime dispatch path, and the CLI flag — the
> parts that produced the negative result + accumulating review
> findings.
>
> Next lever: **B3 — true mixed batching** (decode+prefill packed in one
> C++ entry, plan §3f Option B). The 745 → 5 000 tok/s headroom argument
> in the original B2 plan still stands; what failed was the budget-split
> shape, not the packed-prefill primitive itself.

## Goal

- **optimization** (intended) → **diagnosis** (actual): wire the new
  `qwen35_compiled_prefill_batch_packed` C++ entry into the Metal
  scheduler + runtime so multiple Qwen3.5 prefill rows can pack into one
  MLX forward per tick. Acceptance gate: warm TTFT p99 ≤ 4 s on W3.
  Reality: warm p99 = 30.8 s, no improvement over B1.

## Hypothesis

- Per the plan §1: cold p50 = 22 s for 1024-token prompts at c=16 →
  effective system prefill ≈ 745 tok/s vs mlx-lm Qwen3.5-4B M4 Pro
  ceiling ≈ 5,000 tok/s. Roughly 7× headroom on cold/tool-call TTFT, so
  packing 4 rows per tick should net a meaningful (≥3×) cold-TTFT
  reduction.
- Hypothesis was **falsified** by the bench. Two compounding causes
  identified post-hoc — see Learnings.

## Command

Server:

```bash
cargo build --release -p infer --no-default-features --features metal --bin metal_serve
ln -sfn Qwen3.5-4B-MLX-4bit models/default
RUST_LOG=info target/release/metal_serve \
  --model-path models/default --port 8000 --bind 127.0.0.1 --warmup 1 \
  --metal-max-prefill-rows 4
```

(Note: `--metal-max-prefill-rows 4` is now opt-in. The default is **1**,
which preserves B1 behaviour. The bench below was run with `4` to
exercise the multi-prefill path the commit ships.)

P1 regression-check sweep:

```bash
bash scripts/bench_guidellm.sh metal-b2-c3-multi-prefill \
  --concurrencies 1,2,4,8 --max-seconds 90 \
  --data 'prompt_tokens=256,prompt_tokens_stdev=1,prompt_tokens_min=256,prompt_tokens_max=256,output_tokens=40,output_tokens_stdev=1,output_tokens_min=40,output_tokens_max=40' \
  --processor models/default --model default
```

W3 trace replay (the gate-decider):

```bash
python3 scripts/bench_agent_trace.py \
  --workload agent-w3-short-multiturn --server http://localhost:8000 \
  --label metal-b2-c3-multi-prefill-w3-fix \
  --out bench-output/2026-05-03-metal-b2-c3-multi-prefill-w3-fix/results.json \
  --trace-out bench-output/2026-05-03-metal-b2-c3-multi-prefill-w3-fix/trace.jsonl
```

## Environment

- **Workload:** P1 = guidellm 256-in/40-out concurrent sweep; P2 = `agent-w3-short-multiturn`.
- **Backend:** arle-metal (`--features metal`).
- **Model:** Qwen3.5-4B-MLX-4bit (`mlx-community/Qwen3.5-4B-MLX-4bit`, snapshot `32f3e8ec`).
- **Hardware:** Apple M4 Pro · 48GB unified memory · macOS · Metal · MLX-4bit (macOS 26.3.1, macOSX SDK 26.4, mlx-sys 0.3.0).
- **Commit:** B2 c3 diff on top of `a896c31` (B2 c2). Uncommitted.
- **Feature set:** `cargo build --release -p infer --no-default-features --features metal --bin metal_serve`.
- **`max_prefill_rows`:** 4 for the bench; **default 1** (OFF) in the shipped diff.
- **Pool capacity:** unchanged from B1 (`METAL_PREFIX_POOL_MULTIPLIER = 64`).

## Results

### P1 — guidellm 256-in/40-out sweep (regression check)

| C | TTFT p50 | TTFT p99 | ITL p50 | ITL p95 | TPOT mean | out tok/s | req/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 |    370 |    400 | 13.3 | 13.6 |  22.2 | 45.3 | 1.122 |
| 2 |    484 |    946 | 23.8 | 26.8 |  35.3 | 56.6 | 1.400 |
| 4 |  1,500 |  1,794 | 24.0 | 53.3 |  61.8 | 65.1 | 1.600 |
| 8 |  3,554 |  4,093 | 33.6 | 52.3 | 120.6 | 65.6 | 1.611 |

Δ vs B2 c1 baseline:

| C | metric | B2 c1 | B2 c3 | Δ |
|---|---|---:|---:|---:|
| 1 | TTFT p50 (ms) | 350   | 370   | +5.7% |
| 1 | out tok/s | 47.2  | 45.3  | −4.0% |
| 4 | TTFT p50 (ms) | 776   | 1,500 | **+93.3%** |
| 4 | TTFT p99 (ms) | 1,166 | 1,794 | +53.9% |
| 4 | out tok/s | 68.0  | 65.1  | −4.3% |
| 8 | TTFT p50 (ms) | 3,044 | 3,554 | +16.8% |
| 8 | out tok/s | 68.4  | 65.6  | −4.1% |

P1 regression (synthetic prompts with no prefix sharing) is the
expected first-order effect of dividing the per-tick token budget across
N concurrent prefill rows: each row's chunk shrinks (e.g., from 512 to
128 at N=4), so a 256-token prompt that previously fit in 1 tick now
takes 2.

### P2 — W3 trace replay

| metric | value |
|---|---:|
| successful scored turns | 320 / 320 |
| incomplete scored turns | 0 |
| scored tokens | 20,478 |
| sum scored wall (s) | 8,198 |
| TTFT p50 / p99 (ms) | 25,177 / 30,832 |
| ITL p50 / p99 (ms) | 25.0 / 25.9 |

W3 warm/cold split:

| metric | warm (n=256) | cold (n=64) |
|---|---:|---:|
| TTFT p50 (ms) | 20,643 | 27,004 |
| TTFT p99 (ms) | 30,832 | 29,288 |

W3 warm TTFT by turn_idx (p50 / p99 ms):

| turn_idx | n | TTFT p50 | TTFT p99 |
|---|---:|---:|---:|
| 2 | 64 | 18,408 | 30,522 |
| 4 | 64 | 17,751 | 27,358 |
| 6 | 64 | 20,324 | 29,437 |
| 8 | 64 | 28,272 | 31,184 |

W3 service-side cache / scheduler:

| metric | value |
|---|---:|
| `prefix_hit_rate` (peak) | 27.0% |
| `prefix_skip_rate` (peak) | 26.0% |
| `prefix_request_hit_rate` | 100% |
| `session_affinity_hit / _miss` | 104 / 280 |
| `metal_decode=batch:N/M` | 6325 / 24192 ≈ 26% packed |
| peak active / running_batch | 16 / 16 |
| `peak_mem` | 15.8 GB |
| `cache_mem` | 0.5 GB |

### Δ vs B1 baseline (`2026-05-03-bench-metal-qwen35-prefix-publish-fix.md`)

| metric | B1 (post P1+P2) | B2 c3 (max_rows=4) | Δ |
|---|---:|---:|---:|
| W3 wall total (s) | 7,719 | 8,198 | **+6.2%** |
| W3 warm TTFT p50 (ms) | 20,215 | 20,643 | +2.1% |
| W3 warm TTFT p99 (ms) | 29,380 | 30,832 | +4.9% |
| W3 cold TTFT p50 (ms) | 22,201 | 27,004 | **+21.6%** |
| W3 cold TTFT p99 (ms) | 23,327 | 29,288 | +25.6% |
| W3 prefix_hit_rate | 30.4% | 27.0% | −3.4 pp |
| W3 peak_mem (GB) | 15.5 | 15.8 | +0.3 |
| W3 cache_mem (GB) | 1.0 | 0.5 | −0.5 |

Per-turn warm TTFT B2 c3 vs B1 (p50, ms):

| turn_idx | B1 | B2 c3 | Δ |
|---|---:|---:|---:|
| 2 | 16,682 | 18,408 | +10.3% |
| 4 | 16,857 | 17,751 |  +5.3% |
| 6 | 19,339 | 20,324 |  +5.1% |
| 8 | 27,277 | 28,272 |  +3.6% |

### Acceptance gates

| gate | target | measured | status |
|---|---:|---:|---|
| W3 warm TTFT p99 (ms) | ≤ 4,000 | 30,832 | **MISS** |
| W3 prefix_hit_rate | ≥ 0.7 | 0.27 | **MISS** (arithmetic ceiling 0.67 — independent of B2) |
| P1 c=1,2,4,8 sweep regression | ±5% | up to +93% (c=4 TTFT p50) | **MISS** |
| `cargo test --release` lib | ≥ 614 | 615 | PASS |
| `cargo clippy -- -D warnings` | clean | clean | PASS |

Raw artefacts:

- `bench-output/2026-05-03-metal-b2-c3-multi-prefill/{benchmarks.json,headline_table.md,service_stats_*}` (P1 sweep).
- `bench-output/2026-05-03-metal-b2-c3-multi-prefill-w3-fix/{results.json,trace.jsonl,service_stats_*}` (W3 with the in-flight-fix patch).
- `bench-output/2026-05-03-metal-b2-c3-multi-prefill-w3/results.json` (W3 with the FIRST attempt that aborted 154/384 turns due to the stale-chunk bug — kept for the post-mortem record).
- Server logs: `bench-output/server-logs/2026-05-03T15-*-port8000-metal-b2c3*.log`.

## Problems

- **First W3 run aborted 154/384 turns** with `prefill_qwen35_packed_batch
  chunk_len 255 exceeds row 0 remaining 104` errors. Root cause: the
  scheduler emits each row's chunk based on the runtime snapshot taken
  at the top of the tick, but the B1 prefix-cache import inside
  `activate_pending_request` advances `prompt_cursor` between snapshot
  and dispatch. The packed C++ entry's per-row check then explodes on
  rows whose actual `remaining < scheduled chunk_len`. The singleton
  path handled this gracefully via `min(budget, remaining)` clamp; the
  packed path didn't. **Fixed** in `try_prefill_packed_batch` by
  detecting the mismatch and returning `Ok(None)` (the dispatcher demotes
  the whole group to singleton).

- **Acceptance gate MISS (warm TTFT p99).** B2 c3 with
  `max_prefill_rows=4` did not move warm p99 (30.8 s vs B1 29.4 s), and
  cold TTFT p50 regressed +21.6 %. The plan's "7× headroom" thesis from
  §1 (745 → 5000 tok/s) does not survive contact with the M4 Pro packed
  prefill's actual scaling at batch=2-4: any GPU-batch efficiency
  improvement is more than offset by the per-row chunk shrinking from
  ~256 tok/tick (singleton) to ~64 tok/tick (4-row split), quadrupling
  the per-row tick count.

- **Acceptance gate MISS (P1 regression).** Synthetic 256-in/40-out at
  c=4 saw TTFT p50 jump from 776 ms to 1,500 ms (+93 %). Same root
  cause as W3: per-row chunk shrinkage at the divided budget.

- **Default flipped to `max_prefill_rows = 1` (OFF).** Out of B2 c3 as
  shipped, multi-prefill is opt-in via `--metal-max-prefill-rows N`.
  This preserves B1 behaviour (zero regression in the default
  configuration) while keeping the C++ entry, scheduler config, and
  runtime dispatch in place for B2.1 / B3 follow-ups. The user decides
  whether to ship as opt-in scaffolding or revert outright.

## Learnings

- **"Divide tick budget across N rows" is the wrong policy when chunk_len
  cost is sub-linear in batch size but per-row tick count grows linearly
  in N.** On M4 Pro at chunk_len 64-256 and batch 2-4, the packed-prefill
  forward time scales ~linearly with batch (each row independently does
  attn + MLP), so 4-way batching at 1/4 chunk gives the SAME GPU work
  per tick as singleton at full chunk — but spreads it across 4× more
  ticks per row. The intended win requires either (a) keeping per-row
  chunk_len at the singleton budget and widening total batch tokens
  4× (B2.1 below), or (b) keeping tick budget fixed but ensuring
  packed-prefill is sub-linear on this hardware (Xcode Metal capture
  needed to confirm/refute).
- **Packed-prefill grouping by `chunk_len` is too sparse under W3.**
  After the stale-chunk fix, rows with prefix-cache-advanced cursors
  end up with EFFECTIVELY shorter chunks than the scheduler-emitted
  length, demoting whole groups to singleton. The actual packed-call
  rate is much lower than `max_prefill_rows = 4` would suggest. A
  better grouping (re-bucket by *effective* chunk_len after the cursor
  advance) would lift the engagement rate but adds complexity.
- **Stale-chunk bug exists because scheduler+runtime use different
  cursor-of-record.** Scheduler reads the snapshot's `prompt_progress`;
  runtime reads `prompt_cursor` post-prefix-import. The singleton path
  clamps; the packed path bails. A future architectural cleanup could
  unify the cursor-of-record so the scheduler emits chunks against the
  same value the runtime will see.
- **Memory cost is fine.** B2 c3 peak_mem = 15.8 GB, comparable to B1
  15.5 GB. The packed batch's transient KV+GDR working set is cleared
  per-tick. No memory-related failure mode.

## Δ vs baseline

(See above — explicit Δ table is the headline of this entry.)

## Notes

- Code change scope: 6 files.
  - `infer/src/backend/metal/scheduler.rs` — `MetalSchedulerConfig::max_prefill_rows` (default 1), `build_prefill_rows` rewritten to scan ≥ 0 candidates, new `find_prefilling_requests` plural picker.
  - `infer/src/backend/metal/runtime.rs` — `guard_schedule_step` dispatches via `guard_prefill_packed_batch` for N≥1 prefill rows; new `execute_prefill_packed_batch` with once-per-tick session drain (`drain_qwen35_cpp_sessions_outside`); `dispatch_packed_prefill_group` groups same-`chunk_len` rows for one packed call and demotes the residue + DFlash rows to singleton.
  - `infer/src/backend/metal/request_state.rs` — new `MetalRequestState::try_prefill_packed_batch` (with stale-chunk demote-to-Ok(None)) backed by free function `prefill_qwen35_packed_batch` mirroring `decode_qwen35_packed_batch`'s left-pad + per-row RoPE + per-row sample contract.
  - `infer/src/backend/metal.rs` — `MetalBackendOptions::max_prefill_rows: Option<usize>` field + `MetalBackend::max_prefill_rows_override()` accessor.
  - `infer/src/bin/metal_serve.rs` — `--metal-max-prefill-rows` CLI flag plumbed through `MetalBackendOptions`.
  - `infer/src/bin/metal_bench.rs`, `infer/src/bin/metal_request.rs` — additive `max_prefill_rows: None` in `MetalBackendOptions` initializers (plumbing-only).
- Verify gates met in the default configuration (`max_prefill_rows=1`):
  `cargo fmt --check`; `cargo clippy --release -p infer --no-default-features --features metal -- -D warnings`; `cargo test --release -p infer --no-default-features --features metal --lib` (615 / 0 / 24).
- `e2e_qwen35` test suite has 0 runnable tests on this Mac (model-gated).
- All transient diagnostics removed before this entry.
- Commit body must declare:
  - The default flip (`max_prefill_rows: 4 → 1`) and why.
  - The stale-chunk fix in `try_prefill_packed_batch`.
  - The 6-file scope (over the plan's "≤4 files" because the CLI plumbing required it; the user pre-approved this in the commit-3 brief).
- Codex review pending.

## Codex review v3 + follow-up fixes

Two P1 silent-correctness bugs on the multi-prefill packed path. Both
only manifest with `--metal-max-prefill-rows N` (the default 1 path
never reaches them). Fixed before ship; targeted re-validation recorded.

**P1-1 — packed terminal row dropped its first decoded token.**
`runtime.rs:2303-2308` (the `Ok(Some(results))` branch of
`dispatch_packed_prefill_group`) recorded the sampled token in
`request_state` via `try_prefill_packed_batch` but never called
`ActiveMetalRequest::process_token` like the singleton path does
(see the wrapper at `runtime.rs:153-161`). The model continued from
the token via `last_token`, but the streaming client, incremental
decoder, and pending-token-IDs queue never saw it, so every packed
terminal row would silently drop or corrupt the first generated token.

Fix: after the packed call returns and per-row terminal sampling has
placed the token in `request_state`, iterate the rows whose
`emitted_token` is `Some` and call `request.process_token(token)`
before the snapshot publish + finalize. On `process_token` failure,
record the request as failed and cancel the row (same shape as the
singleton path).

**P1-2 — packed dispatch left non-group sessions live.**
`runtime.rs:2125-2126` passed the packed group's `row_ids` to
`drain_qwen35_cpp_sessions_outside`, which skipped them. If a tick
contained a packed group AND a singleton-residue prefill row with a
different `chunk_len`, the residue row's active C++ session stayed
live — and the packed entry's `begin_session` then races the bridge's
single-session invariant.

Fix: pass `&[]` (drain EVERY active Qwen3.5 C++ session, including the
packed group's own rows). The packed group re-loads its KV/GDR into
the bridge inside `prefill_qwen35_packed_batch`, so the redrain is
correctness-neutral for them. Smallest-diff option codex offered;
takes one extra pair of `end_session` / `begin_session` calls per
group on the packed-path-enabled tick.

**Targeted re-validation (server started with `--metal-max-prefill-rows 4`):**

- 8-turn aligned same-session diag (sequential, c=1): turn 0 cold TTFT
  = 1,590 ms; turns 1–7 warm TTFT = 92–96 ms each (B1-equivalent).
- 6-turn cold-vs-warm SHA-256 byte-equality test on a deliberately
  misaligned prompt length: byte-identical for every turn (cold then
  warm via cache vs all-warm via cache).
- 4-request concurrent vs 4-request sequential test
  (`/tmp/track_a_diag4.py`): SHA-256 byte-identical replies across
  every request slot. This actually exercises the packed path —
  4 requests submitted simultaneously trigger
  `dispatch_packed_prefill_group` with 4 rows.
- Server log: zero `ERROR` / `panic` / `prefill session already
  active` / `process_token failed` lines under any of the above runs.

The W3 acceptance gate remains MISSED for the reasons in §Problems —
the silent-correctness fixes don't change the per-row budget split
that drives the cold-TTFT regression. The fixes only restore the
multi-prefill path to a state where its (still net-regressing)
behavior is at least observable + correct.

## Recommended follow-ups (none of these are this commit)

- **B2.1 — per-row budget widening.** Drop the "divide across N rows"
  policy; give each prefill row up to `max_batch_tokens` (or a
  per-row cap like `max_batch_tokens / 2`). Total batch tokens scales
  with `max_prefill_rows`. Memory cost is bounded by
  `max_prefill_rows × max_batch_tokens × per-token-bytes`. Re-bench
  W3; if cold TTFT improves AND the per-row chunk_count stays bounded,
  ship as the actual win.
- **B2.2 — re-group by effective chunk_len.** After the stale-chunk
  cursor advance, rebucket rows by their EFFECTIVE chunk_len. Increases
  packed-call engagement without changing the policy. Pairs well with
  B2.1.
- **B2.5 — chunked prefill per the plan §11 lever 2.** Split a single
  >budget prompt across multiple ticks at fixed chunk size N<512. The
  *latency* lever (helps tail TTFT under contention) but does not raise
  *throughput*. Prerequisite if B2.1 alone doesn't hit ≤ 4 s.
- **B3 — true mixed batching (Option B per plan §3f).** Pack decode
  (`seq_len = 1`) and prefill rows (`seq_len = chunk_len`) into ONE
  batched C++ call with varlen query length. Bigger surface (new C++
  entry + mask/RoPE generalization) but unblocks the cold-TTFT floor
  by removing the "decode then prefill" tick boundary.
- **Cold-TTFT diagnostic.** Run an Xcode Metal capture against
  `qwen35_compiled_prefill_batch_packed` at batch=1 vs 4 to measure
  actual GPU efficiency scaling. The B2 c3 W3 result strongly suggests
  the M4 Pro packed forward at batch 2-4 is approximately linear in
  total tokens (not sub-linear), which would invalidate the "7×
  headroom" hypothesis from plan §1 on this hardware.
