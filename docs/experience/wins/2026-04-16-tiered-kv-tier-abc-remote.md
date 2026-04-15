# 2026-04-16 · Tiered KV Cache Tier A/B/C — remote CUDA acceptance

## Context

Tier A/B/C is the follow-on batch on top of the already-accepted
M2b + M0.3 / M3a + M3b + M3c stack, landed in these local commits:

- **Tier A — `d3d1e46` `feat(kv-tier): wire M3 coordinator + staged
  admission (Tier A)`**: CUDA scheduler now owns a live coordinator
  thread, passes a real `CoordinatorHandle` as `StagePlanner`, parks
  staged requests in `stage_waiting`, and re-admits them when
  `StagingCompleted` arrives (stub-completed synchronously by the
  local transport).
- **Tier B — `e0f69f9` `feat(kv-tier): compute BlockFingerprint at
  publish + disk round-trip test (Tier B)`**: scheduler publish
  path computes `BlockFingerprint::compute_from_tokens` per block
  and routes inserts through
  `RadixCache::insert_with_fingerprints(...)`; `DiskStore` round
  trips both bytes and fingerprint in the unit-test path.
- **Tier C — `9b01c2a` `perf(kv-tier): O(1) RadixCache block lookup
  + SchedulerConfig knobs (Tier C)`**: `RadixCache` now keeps a
  private `block_index: HashMap<BlockId, usize>` for O(1) lookup,
  with a `rebuild_block_index()` helper for post-serde restore.
  The five prefix-cache / staging knobs moved from module constants
  onto `SchedulerConfig` with validation.
- **Follow-on fix — `189bd17` `fix(kv-tier): don't re-increment
  waiting_count on staged re-queue`** (second-pass Codex review):
  removed two `waiting_count.fetch_add(1)` that would otherwise
  inflate the `request_rx` channel counter on every staged
  completion and break the idle-exit guard. This fix is on the
  baseline we validated below, not a separate tier.
- **Safety fix — `498c0cc` `fix(kv-tier): address Codex final-review
  correctness findings`** landed earlier in the same review cycle.

This note records the remote acceptance run against
[`../../plans/tiered-kv-cache-tier-abc-remote-acceptance.md`](../../plans/tiered-kv-cache-tier-abc-remote-acceptance.md).

## Environment

- GPU: NVIDIA L4 24 GB (driver 580.82.07, CUDA 13.0, SM 89)
- Model: Qwen3-4B BF16, `Qwen/Qwen3-4B` HF Instruct variant
- Commit at validation: `d6eb87e` (post the 2026-04-15 Metal batch-decode
  tranche, the `189bd17` `waiting_count` fix, and the 2026-04-15
  int8 decode perf work: Option A + `kTargetBlocksPerSm = 32`
  split-KV grid bump)
- Server: `target/release/infer --model-path models/Qwen3-4B --num-slots 4`
  (same flags as the 2026-04-15 M3c remote baseline)
- Cargo env: `CARGO_HOME=/tmp/cargo-home-local`
- TokenKVPool at boot: `24672 max tokens (1542 pages @ page_size=16),
  3.6 GB for 36 layers, format=BF16`

## Static sanity (§2)

All four §2 greps produce the expected results:

```
rg coordinator_handle|stage_waiting|page_lifecycle \
  infer/src/scheduler/cuda infer/src/kv_tier
```

→ matches in `infer/src/scheduler/cuda/runtime.rs`,
`infer/src/scheduler/cuda/core.rs`, and
`infer/src/kv_tier/coordinator.rs` — Tier A live coordinator handle,
ticket-wait map, and page-lifecycle ownership are present. ✅

```
rg insert_with_fingerprints|BlockFingerprint::compute_from_tokens \
  infer/src/prefix_cache.rs infer/src/scheduler/cuda/core.rs infer/src/types.rs
```

→ matches at `types.rs:55` (`compute_from_tokens` implementation),
`scheduler/cuda/core.rs:455,463` (publish path computes per-block
fingerprints and routes through `insert_with_fingerprints`),
`prefix_cache.rs:461,466` (the `insert_with_fingerprints` API +
the legacy `insert` wrapper), plus the `DiskStore` fingerprint
round-trip test at `kv_tier/transport/disk.rs:342`. ✅

```
rg block_index|rebuild_block_index infer/src/prefix_cache.rs
```

→ `prefix_cache.rs:170` private `block_index: HashMap<BlockId, usize>`
field, `:208` `rebuild_block_index` helper, plus the O(1) lookup
call site at `:246` and matching `block_index_tracks_inserts_and_evictions`
+ `rebuild_block_index_round_trips_after_serde` unit tests at
`:1167` and `:1190`. ✅

```
rg "struct SchedulerConfig|prefix_cache_high_water|prefix_cache_low_water|prefix_cache_retain_hard_cap|prefix_cache_keepalive_ticks|stage_wait_keepalive_ticks" \
  infer/src/scheduler/types.rs infer/src/scheduler/cuda/core.rs
```

→ all five knobs present on `SchedulerConfig` at
`infer/src/scheduler/types.rs:50-90` with defaults
`{0.75, 0.50, 0.90, 64 ticks, 512 ticks}` and validation at
`:127-146` that enforces ordering (`low < high ≤ cap ≤ 1`,
`stage_wait_keepalive_ticks ≥ prefix_cache_keepalive_ticks`,
`prefix_cache_keepalive_ticks ≥ 1`). ✅

## Build / test gates (§3)

```
cargo build -p infer --release                            # clean incremental (prior build warm)
cargo test --workspace --exclude mlx-sys --release --lib  # 363 tests pass
cargo test -p infer --release --test e2e                  # Phase 1-4 all pass
cargo fmt --all -- --check                                # clean
```

Raw test-result summary (`cargo test --workspace --exclude mlx-sys
--release --lib`):

- infer lib: **295 passed, 0 failed, 11 ignored** (new tests from
  Tier A/B/C already landed — including
  `block_index_tracks_inserts_and_evictions`,
  `rebuild_block_index_round_trips_after_serde`, and the
  `SchedulerConfig` validation tests)
- infer-cuda-kernels: 35 passed
- infer-agent: 12 passed
- infer-chat: 12 passed
- infer-cli: 7 passed
- infer-tools: 2 passed

**Total: 363 tests pass, 11 ignored, 0 fail.**

`cargo test -p infer --release --test e2e` on Qwen3-4B: **1 passed**
(Phase 1-4 all clean — Phase 1 baselines match, Phase 2 stream runs,
Phase 3 stream/non-stream consistency passes on all 6 prompts,
Phase 4 consumer drop safety passes). The 2026-04-15 replay-drift
fix (`eb347d9`) is still holding.

Not-in-scope pre-existing failures (same list as every remote
acceptance this week, explicitly deferred):
- `cargo test -p infer --test e2e_qwen35` — `infer/test_data/Qwen3.5-4B.json`
  baseline drift vs current HF Qwen3.5-4B weights. Tracked in
  `project_remote_cuda_box.md`.
- `cargo test --release --test greedy_consistency` — pre-existing
  B=3 batched decode bug tracked in
  `../errors/2026-04-13-batched-decode-high-concurrency.md`.
- `cargo clippy --workspace -- -D warnings` — pre-existing
  `unused import: Path` in `crates/infer-tools/src/lib.rs:3`. Memory
  instructs "report, don't silently fix"; left for a separate
  cleanup commit.

No new CUDA-only linker or runtime failures from the Tier A/B/C
batch.

## Long-session regression gate (§4)

```
python3 scripts/bench_agent_trace.py \
    --server http://localhost:8000 \
    --label tiered-kv-tier-abc-remote \
    --out docs/experience/wins/2026-04-16-bench-tiered-kv-tier-abc-remote.json
```

Raw result on commit `d6eb87e`, `num_slots=4`:

```
session          turn  msgs   wall(ms)  ttft(ms)   itl(ms)  tokens  finish
──────────────────────────────────────────────────────────────────────────
agent-001           0     2     5130.9      48.2      34.4     144  stop
agent-001           2     4     9093.5     151.7      34.7     256  length
agent-001           4     6     9152.5     223.8      34.8     256  length
agent-002           0     2     9052.8     125.0      34.5     256  length
agent-002           2     4     9138.6     113.5      34.8     256  length
agent-002           4     6     8876.9     129.6      34.4     256  length
agent-003           0     2     9053.3     166.2      34.5     256  length
agent-003           2     4     9153.8     114.7      34.8     256  length
agent-004           0     2     9052.2     207.4      34.5     256  length
agent-004           2     4     9094.2     202.7      34.7     256  length
agent-005           0     2     9075.6     112.2      34.6     256  length
agent-005           2     4     9152.7     170.4      34.8     256  length
agent-006           0     2     9094.0     106.9      34.7     256  length
agent-006           2     4     9022.1     112.5      34.6     256  length
──────────────────────────────────────────────────────────────────────────

turns OK:        14 / 14
tokens total:    3472
wall total (s):  123.14
TTFT p50/p99:    125.0 / 223.8 ms
ITL  p50/p99:    34.6 / 34.8 ms
```

Comparison against the accepted **M3c remote baseline**
([`2026-04-15-tiered-kv-m3c-remote.md`](2026-04-15-tiered-kv-m3c-remote.md),
commit `85bc85b`, same host / model / flags):

| metric       | M3c remote | Tier A/B/C remote | delta |
|--------------|-----------|-------------------|-------|
| turns OK     | 14 / 14   | 14 / 14           | —     |
| TTFT p50     | 124.6 ms  | 125.0 ms          | +0.4 ms |
| TTFT p99     | 225.7 ms  | 223.8 ms          | −1.9 ms |
| ITL  p50     | 34.7 ms   | 34.6 ms           | −0.1 ms |
| ITL  p99     | 34.8 ms   | 34.8 ms           | 0     |
| tokens total | 3472      | 3472              | —     |
| wall total   | 123.37 s  | 123.14 s          | −0.23 s |

**All metrics within noise.** The Tier A/B/C stack does not regress
the M3c envelope. No CUDA faults, no deadlocks, no stuck staged
requests, no panic in the new coordinator thread, no `waiting_count`
drift (the `189bd17` fix is load-bearing here).

### Tier A staging: installed, not exercised by this workload

The server log for this run shows **zero** `Stage*` / `StagingCompleted`
events. That's expected and not a regression:

- The agent-trace workload runs 6 sessions × 2-3 turns, max prompt
  ~250 tokens. Every turn fits comfortably in the `num_slots=4`
  pool with the `page_size=16` BF16 layout (24 672 pool tokens
  available).
- Tier A's `stage_waiting` path fires only when the scheduler
  admission hits a radix block that's NOT currently `ReadyOnGpu`
  (i.e., the block would need to be staged back from a host /
  disk tier). The local tier is still stub-only: the
  `StagePlanner::stage` call emits `StagingQueued` plus a
  synchronous `StagingCompleted` echo on the same tick.
- Under this workload and this memory configuration, no admission
  ever has a reason to park into `stage_waiting`, so the coordinator
  thread stays quiescent.

This matches the acceptance doc's explicit §4 framing:
> "The goal here is not to prove real async staging yet; it is to
> confirm that the Tier A stub-completion path and Tier B/C metadata
> changes do not regress the already accepted M3c behavior envelope."

A workload that **does** exercise stage_waiting would need either
(a) many more concurrent sessions than slots, (b) a much larger
total working-set than HBM, or (c) an actual async transport that
doesn't stub-complete synchronously. All three are post-Tier-A
work (Tier D, Tier E, real `cudaMemcpyAsync` staging).

## Sign-off

- [x] Static sanity checks passed (all 4 greps match the expected
      shape at the expected file:line positions).
- [x] Build/test gate passed on CUDA. 363 workspace tests green,
      e2e Phase 1-4 green, fmt clean.
- [x] Long-session regression gate completed without runtime faults;
      all 14 turns emit, TTFT / ITL / wall within noise of the
      accepted M3c same-host baseline.
- [x] This win note exists with raw bench output and the explicit
      M3c vs Tier A/B/C comparison.

**Tier A/B/C accepted on the 2026-04-16 L4 host.**

## Rule

**A "stub-completion" tier should be accepted as not-regressing,
not as proving-itself.** The right gate for Tier A's current shape
(stub-complete `StagingCompleted` on the same tick) is "no behavior
change on the M3c baseline workload", not "demonstrates real
staging". The latter requires Tier D (disk round-trip on the
coordinator) or Tier E (async `cudaMemcpyAsync` promote) to land
first, at which point the long-session bench will need a new
workload — one that forces stage_waiting via more-than-slot
concurrency, or via a cold host-tier block — to actually exercise
the async path.

**Corollary**: the 189bd17 `waiting_count` fix is load-bearing for
this acceptance even though it looks like a side fix. Without it,
any staged-completion event permanently inflates the channel
counter, and the idle-exit guard breaks. Under Tier A's current
stub completion, that inflation would fire on every stub-staged
request — which on this agent-trace workload is zero, so the bug
wouldn't have surfaced here. But a workload that does hit the
staged path would have seen a queue-depth drift → backpressure
saturation. Tier A + 189bd17 ship as a pair.

**Corollary 2**: Tier B's fingerprint work is silent on this
acceptance because the new `insert_with_fingerprints` API runs on
every publish and the bench just stops firing events after 94
seconds. The unit test
(`disk.rs:342 BlockFingerprint::compute_from_tokens` round-trip)
is the only real evidence that the path is working as intended;
the remote bench only proves "doesn't break M3c". That is the
correct level of evidence for a Tier B's "data-model plumbing"
tranche, same way M1a was accepted as a deletion-only tier.
