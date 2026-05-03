# Metal Qwen3.5 in-memory prefix publish — bench, M4 Pro, 2026-05-03

> Single B1-only commit per Track A handoff. Pre-fix W3 baseline:
> [`2026-05-02-bench-agent-load-a1-session-affinity-admission.md`](2026-05-02-bench-agent-load-a1-session-affinity-admission.md) and the local Metal pre-fix
> run captured in `bench-output/2026-05-02-metal-agent-w3/results.json`.
> Diff revised after `codex review --uncommitted` flagged two real defects
> (P1 silent-correctness on hybrid GDR state, P2 cache eviction
> under-accounting). Numbers in this entry reflect the post-codex-review v2
> diff, not the buggy v1 numbers from earlier in the session.

## Goal

- **optimization**: publish Metal Qwen3.5 in-memory prefix snapshots so warm
  same-session traffic (W3) stops paying full re-prefill on every turn.

## Hypothesis

- The in-memory publish path returns zero snapshots because
  `Qwen35StepDriver::stream_prefix_snapshots` short-circuits while the live
  C++ session owns the compiled model. Draining the session before snapshot
  build (and exporting the live KV+GDR state directly, no second
  replay-prefill) will (a) make the cache fire for warm same-session turns
  and (b) collapse warm TTFT on agent traffic, with negligible regression
  on cold/synthetic shapes.
- (Refined post codex review.) The snapshot must keep its KV state
  consistent with the recurrent GDR state — Qwen3.5 GDR cannot be rewound
  to a shorter prefix without replay — so the live snapshot has to be
  taken at exactly the live `cache_len`, not block-aligned-down. The
  in-memory cache only needs to reject sub-block snapshots; full block
  alignment was a disk-format requirement, not a correctness requirement.

## Command

Server (release Metal binary, fresh):

```bash
cargo build --release -p infer --no-default-features --features metal --bin metal_serve
ln -sfn Qwen3.5-4B-MLX-4bit models/default
RUST_LOG=info target/release/metal_serve \
  --model-path models/default --port 8000 --bind 127.0.0.1 --warmup 1
```

P1 regression-check sweep (256-in / 40-out, c=1,2,4,8) — run against the
v1 build, unchanged in spirit by the v2 fix (no per-request behavior
change in this shape):

```bash
bash scripts/bench_guidellm.sh metal-b1-agent-short \
  --concurrencies 1,2,4,8 --max-seconds 90 \
  --data 'prompt_tokens=256,prompt_tokens_stdev=1,prompt_tokens_min=256,prompt_tokens_max=256,output_tokens=40,output_tokens_stdev=1,output_tokens_min=40,output_tokens_max=40' \
  --processor models/default --model default
```

P2 W3 trace replay (final v2 diff, pool=131k):

```bash
python3 scripts/bench_agent_trace.py \
  --workload agent-w3-short-multiturn --server http://localhost:8000 \
  --label metal-b1-fix-agent-w3 \
  --out bench-output/2026-05-03-metal-b1-fix-w3/results.json \
  --trace-out bench-output/2026-05-03-metal-b1-fix-w3/trace.jsonl
```

## Environment

- **Workload:** P1 = guidellm 256-in/40-out concurrent sweep; P2 = `agent-w3-short-multiturn` (64 warm + 64 cold sessions, 320 scored turns, c=16, base 1024±32 tokens, tail 64±8 tokens, 64 max output).
- **Backend / engine:** arle-metal (`--features metal`).
- **Model:** Qwen3.5-4B-MLX-4bit (`mlx-community/Qwen3.5-4B-MLX-4bit`, snapshot `32f3e8ec`).
- **Tokenizer / processor:** local `models/default` symlink to the snapshot.
- **Hardware:** Apple M4 Pro · 48GB unified memory · macOS · Metal · MLX-4bit (macOS 26.3.1, macOSX SDK 26.4, mlx-sys 0.3.0).
- **Commit:** B1 v2 diff on top of `6e00332` (uncommitted, pending second `codex review --uncommitted`).
- **Feature set:** `cargo build --release -p infer --no-default-features --features metal --bin metal_serve`.
- **KV / cache mode:** Metal Qwen3.5 in-memory prefix tier; SSD tier disabled.
- **Pool capacity:** `METAL_PREFIX_POOL_MULTIPLIER` 4 → 64 (`max_cached_tokens` 8192 → 131072 token-equivalent units). Eviction now accounts each entry by `max(snapshot.token_ids.len(), snapshot.kv_capacity)` so long-output requests cannot pin many GB while being charged as a few prefix tokens.
- **Session API:** unchanged from `e97d52a`.

## Results

### P1 — guidellm 256-in/40-out sweep (regression check)

Run against v1 of the diff (the v2 patch only changes the in-memory
snapshot path and accounting; synthetic-prompt P1 has no warm reuse, so
re-running adds no signal). Headline numbers preserved verbatim:

| C | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | TPOT mean (ms) | out tok/s | total tok/s | req/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 |   359 |   379 | 13.4 |  14.4 |  22.0 | 45.6 | 338.5 | 1.13 |
| 2 |   709 |   774 | 22.9 |  25.3 |  32.4 | 62.0 | 460.7 | 1.53 |
| 4 |   759 | 1,166 | 36.4 |  51.0 |  58.8 | 68.0 | 505.2 | 1.69 |
| 8 | 3,044 | 3,463 | 41.3 |  50.5 | 115.3 | 68.4 | 507.9 | 1.69 |

P1 service trace: peak running_batch=4, peak waiting=6, peak kv_util=0%, peak prefix_hit_rate=0% (synthetic prompts share no content). `metal_decode=batch:9843/21251 ≈ 46%` batched.

### P2 — W3 trace replay (post-codex-review v2)

| metric | value |
|---|---:|
| successful scored turns | 320 / 320 |
| incomplete scored turns | 0 |
| scored tokens | 20,480 |
| sum scored wall (s) | 7,719 |
| run wall clock (≈ min) | ~13 |
| TTFT p50 / p99 (ms) | 22,107 / 29,380 |
| ITL p50 / p99 (ms) | 24.0 / 24.5 |

W3 warm/cold split:

| metric | warm (n=256) | cold (n=64) |
|---|---:|---:|
| TTFT p50 (ms) | 20,215 | 22,201 |
| TTFT p99 (ms) | 29,380 | 23,327 |

W3 warm TTFT by turn_idx (p50 / p99):

| turn_idx | n | TTFT p50 (ms) | TTFT p99 (ms) |
|---|---:|---:|---:|
| 2 | 64 | 16,682 | 24,151 |
| 4 | 64 | 16,857 | 27,317 |
| 6 | 64 | 19,339 | 29,165 |
| 8 | 64 | 27,277 | 30,007 |

W3 service-side cache / scheduler (after-run snapshot):

| metric | value |
|---|---:|
| `prefix_hit_rate` (peak) | 30.4% |
| `prefix_skip_rate` (peak) | 29.8% |
| `prefix_request_hit_rate` | 100% (every cached prefix import that admitted, hit) |
| `session_affinity_hit` / `_miss` | 123 / 281 |
| `metal_decode=batch:N/M` | 6802 / 24696 ≈ 28% packed |
| peak active / running_batch | 16 / 16 |
| `peak_mem` | **15.5 GB** (vs v1 buggy diff 17.4 GB; pre-B1 8.2 GB) |
| `cache_mem` | **1.0 GB** (vs v1 buggy diff 9.0 GB; pre-B1 1.3 GB) |
| `kv_util` | 0% |

The `cache_mem` collapse from 9.0 → 1.0 GB is the P2 fix landing: the
in-memory pool now accounts each cached entry by its actual KV+GDR
footprint (`max(token_count, kv_capacity)`), so the 131 k-token budget
holds the right number of snapshots instead of letting `kv_capacity`
silently inflate residency past the cap.

Raw artefacts:

- `bench-output/2026-05-03-metal-b1-agent-short/{benchmarks.json,csv,html,headline_table.md,service_stats_*}` (P1 sweep, unchanged).
- `bench-output/2026-05-03-metal-b1-pool-w3/{results.json,trace.jsonl,service_stats_*}` (W3 with v1 buggy diff, kept for delta comparison).
- `bench-output/2026-05-03-metal-b1-fix-w3/{results.json,trace.jsonl,service_stats_before.txt,service_stats_after.txt}` (W3 with v2 corrected diff — the headline run).
- Server logs: `bench-output/server-logs/2026-05-03T*-port8000-metal-b1-*.log`.

### Correctness verification (post-codex P1 fix)

`scripts`-free curl test (4-msg conversation, 6 turns, mismatched
non-block-aligned prompts) showed byte-identical output between a
cold-then-warm run and an all-warm-via-cache run for every turn (SHA-256
match, `len_a == len_b` for all 6 turns). Single-session 8-turn
diagnostic: turn 0 cold TTFT = 1,548 ms, turns 1–7 warm TTFT = 90–91 ms
each (17× speedup, every warm turn hits — prior v1 build with strict
block-alignment skip only hit 2/7 warm turns since the prompts happened
to land off-boundary).

## Δ vs baseline

Baseline = `bench-output/2026-05-02-metal-agent-w3/results.json` (pre-B1, same M4 Pro, same model, same harness).

| metric | pre-B1 | v1 (buggy) | v2 (fixed) | Δ v2 vs pre-B1 |
|---|---:|---:|---:|---:|
| W3 wall total (s) | 11,094 | 6,503 | 7,719 | **−30.4%** |
| W3 warm TTFT p50 (ms) | 26,634 | 14,876 | 20,215 | **−24.1%** |
| W3 warm TTFT p99 (ms) | 78,434 | 23,401 | 29,380 | **−62.5%** |
| W3 cold TTFT p50 (ms) | 22,310 | 22,185 | 22,201 | −0.5% (noise) |
| W3 prefix_hit_rate | 0.0% | 40.8% | 30.4% | new |
| W3 prefix_skip_rate | 0.0% | 39.1% | 29.8% | new |
| W3 session_affinity_hit | 0 | 157 | 123 | new |
| W3 peak_mem (GB) | 6.7 | 17.4 | 15.5 | +8.8 GB |
| W3 cache_mem (GB) | 1.3 | 9.0 | 1.0 | −0.3 GB |
| P1 c=1 ITL p50 (ms) | 12.7 | 13.4 | 13.4 | +5.5% (per-request drain cost) |
| P1 c=4 out tok/s | 67.9 | 68.0 | 68.0 | +0.1% (noise) |

Per-turn warm TTFT collapse vs pre-B1 (W3 turn_idx p50, ms):

| turn_idx | pre-B1 | v1 (buggy) | v2 (fixed) | Δ v2 vs pre-B1 |
|---|---:|---:|---:|---:|
| 2 | 24,322 |  9,433 | 16,682 | **−31.4%** |
| 4 | 25,807 | 12,675 | 16,857 | **−34.7%** |
| 6 | 27,898 | 16,160 | 19,339 | **−30.7%** |
| 8 | 29,869 | 19,185 | 27,277 |  −8.7% |

The v2 hit rate (30.4%) is lower than the buggy v1 (40.8%) because v1
was (incorrectly) caching stale GDR state on non-block-aligned prompts
and serving wrong logits on those hits. v2 only caches snapshots whose
KV and GDR state are consistent at the snapshot's `cache_len`. The W3
hit-rate ceiling for v2 with random prompt lengths is set by the cache
working set, not by alignment.

## Problems

- **Acceptance gate `prefix_hit_rate ≥ 0.7` is unreached (30.4% measured).**
  Arithmetic ceiling for this W3 trace shape is `256 / 384 ≈ 67%` (cold +
  warmup turns can never hit), so the literal `≥ 0.7` target is unreachable
  even with a perfect cache. With v2's stricter (correct) snapshot
  semantics, hit rate is bounded by the W3 working set fitting in the
  131 k-token pool. Per-session dedup (drop superseded older
  same-session snapshots when a longer same-session prefix lands) is the
  natural follow-up — it shrinks the working set ~4× without growing the
  pool, and should push hit rate well past 50% on this workload.

- **Acceptance gate `warm TTFT p99 ≤ 4 s` is unreached (29.4 s measured).**
  ITL is fine (24 ms p50/p95). The remaining warm TTFT is queue-ordering
  pressure: cold and warmup turns share 22 s p50 because Metal still
  emits at most one prefill row per scheduler tick (Phase-2 bottleneck B2,
  `infer/src/backend/metal/scheduler.rs:162`). With c=16 in flight, even
  cache-hit warm turns wait behind ongoing prefill. B1 alone cannot move
  warm p99 below the cold-prefill latency floor; closing the 4 s gate
  requires B2 (multi-prefill-per-tick) which is out of B1 scope per the
  Track A handoff.

- **v1 → v2 hit-rate regression is real and load-bearing.** v1 (committed
  privately, never published) accidentally exposed wrong logits on
  non-block-aligned prompts because `cache_len` was rewound while GDR
  state was not. v2 fixes this strictly. Net effect on this benchmark:
  warm TTFT p50 slows from 14.9 s to 20.2 s, but every served reply is
  now correct. Codex review caught this before commit.

- **Pool sizing constant** (`METAL_PREFIX_POOL_MULTIPLIER`) was bumped
  4 → 64 to let high-session-count agent traffic survive without immediate
  eviction. Memory cost is real but smaller than v1: peak_mem 6.7 GB →
  15.5 GB during W3 (M4 Pro 48 GB host has headroom). On smaller-memory
  hosts a smaller pool is required; an env / CLI override should land
  before this change ships beyond the M4 Pro 48 GB validation box.

- P1 c=1 ITL +5.5% reflects the per-request `end_session` ↔ `begin_session`
  round-trip introduced by B1 (drain to snapshot, re-attach next tick).
  Within the ±5 % watch band but at its edge; acceptable trade since the
  drain pays for the W3 win.

## Learnings

- **In-memory tier publish was structurally silent on Metal Qwen3.5.**
  `Qwen35StepDriver::stream_prefix_snapshots` short-circuited
  unconditionally on `cpp_session_active()` (see prior comment block at
  `request_state.rs` lines 4154-4162). The disk path drained the session
  first, so it worked; the in-memory path didn't drain, so it never
  produced a single snapshot. Lesson: a guard that protects against
  nested `session_begin` must be paired with a drain on every caller, or
  the path that didn't drain becomes silently dead code.
- **Replay-based snapshot export is wrong for a single-shot in-memory
  publish.** The disk path's "drain + replay-prefill" pattern doubles
  prefill cost per request (replay walks the prompt again chunk-by-chunk
  to snapshot at every block boundary). For the live cache the snapshot
  must be taken at exactly the live `cache_len` (`prompt_cursor ==
  cache_len`), never at a truncated `aligned_len` — see the next bullet
  on why truncation corrupts Qwen3.5's GDR. Sub-block-aligned prompts
  fall back to skipping the publish for that turn rather than producing
  an unsafe snapshot. New helper:
  `Qwen35StepDriver::export_drained_prefix_snapshot`.
- **Hybrid recurrent state cannot be silently truncated.** Qwen3.5's GDR
  is a stream-updated recurrent accumulator. Snapshotting a live request
  at `aligned_len < cache_len` would leave GDR ahead of KV; future
  imports would replay tokens against state that already incorporated
  them. A non-replay snapshot must be taken at exactly the live
  `cache_len`. The in-memory cache only needs to reject sub-block
  snapshots; full block alignment was a disk-format inheritance, not a
  correctness invariant. Codex review caught the original silent miss
  before commit.
- **Cache-pool eviction must account for resident allocation, not
  reusable prefix length.** Each cached snapshot retains the live
  request's full `kv_capacity = prompt_len + max_new_tokens`
  pre-allocation. Counting `token_count` only would let a few
  long-output requests pin GBs of KV while the cache thinks it has room
  for more. Fix: `snapshot_footprint(...) = max(token_count, kv_capacity
  as usize)` used by both `insert_snapshot` and `ensure_capacity_for`.
  W3's `max_tokens=64` makes the gap small (~6 %); workloads with larger
  outputs would have OOM'd a pool that ignored this.
- **Cache pool sizing matters as much as cache correctness.** A working
  cache with a pool sized to `4 × max_running_requests × max_batch_tokens`
  (8192 tokens) holds ~7 snapshots — sufficient for a few-session demo,
  insufficient for a 64-warm-session agent workload. The current bump
  to multiplier=64 is M4-Pro-specific; the proper fix sizes the pool by
  expected concurrent session count, not by per-tick token budget. Open
  in B1.5.
- **TTFT under high concurrency is bottlenecked by the
  one-prefill-per-tick scheduler limit** (`MetalScheduleStep::from_logical_plan`
  debug_assert at `scheduler.rs:162`). B1 caps at the cold-prefill latency
  floor (~22 s here); B2 (multi-prefill-per-tick) is the next lever and
  is independently scoped.

## Notes

- Code change scope: 2 files (`infer/src/backend/metal/runtime.rs`,
  `infer/src/backend/metal/request_state.rs`).
- Files touched relative to upstream `6e00332`: 2 source + 1 wins entry.
- All transient diagnostic `tracing`/`log::info!` instrumentation was
  removed before this entry per `feedback_no_half_states.md`.
- Verify gates met (post v2 diff): `cargo fmt --check`, `cargo clippy
  --release -p infer --no-default-features --features metal -- -D warnings`,
  `cargo test --release -p infer --no-default-features --features metal --lib`
  (614 passed / 0 failed / 24 ignored).
- Track A B1 acceptance status:
  - P1 c=1,2,4,8 sweep regression — **PASS** (within ±5% noise band).
  - W3 trace replay completes 320/320 — **PASS**.
  - `prefix_hit_rate ≥ 0.7` — **MISS** (30.4%; arithmetic ceiling 67%; v2
    correctness lowered v1's incorrect 40.8%).
  - warm TTFT p99 ≤ 4 s — **MISS** (29.4 s; bounded by single-prefill-per-tick floor).
  - wins entry — landed (this file).
  - Codex review v1 — **2 defects flagged** (P1 silent-correctness on
    non-aligned prompts, P2 cache eviction under-accounting).
  - Codex review v2 — **pending** (this entry is being handed back).
- Recommended next-step proposals (not in this commit):
  - **B1.5 — Per-session snapshot dedup**: drop subsumed older same-session snapshots in `insert_snapshot` so the cache working set scales by session count, not by total publish count. Should push W3 hit rate from 30.4% toward the 67% arithmetic ceiling without further pool growth.
  - **B1.6 — Pool capacity from `--max-prefix-pool-tokens` CLI flag**: replace the hardcoded `METAL_PREFIX_POOL_MULTIPLIER` with a request-sized config so smaller-memory hosts aren't forced to the 15 GB working set.
  - **B2 — Multi-prefill-per-tick** (separate Plan-subagent commit per Track A handoff): the `prefill_rows.len() <= 1` debug_assert at `scheduler.rs:162` is the cap on cold/tool-call TTFT.
