# Tiered KV Cache — execution plan (local / remote-GPU / parallel-GPU)

**Status**: Active execution split for [`../projects/tiered-kv-cache.md`](../projects/tiered-kv-cache.md).

This doc carves every phase from the project plan into three lanes so the
local Mac dev box and the remote CUDA host can both stay busy at all times.
The project doc is the **what + why**; this doc is the **where you do it**.

---

## 0 · Build matrix recap

| Lane | Build invocation | Runs | Does NOT run |
|---|---|---|---|
| **Mac · no-cuda** | `cargo check --no-default-features --features no-cuda` | type/borrow check, non-GPU unit tests | CUDA kernels, FlashInfer link, e2e |
| **Mac · metal** | `cargo build --release --no-default-features --features metal` | Metal backend, `metal_kv_pool` tests, `mlx-sys` bindings | scheduler/cuda/*, FlashInfer |
| **Remote CUDA** | `cargo build --release` (default features = `["cuda"]`) | full stack, `e2e`, `e2e_qwen35`, `greedy_consistency`, `bench_throughput_sweep.py` | n/a |

Lane rules:
- **Local lane** = anything that compiles green under both `--features no-cuda` AND `--features metal`. Touching it never risks the CUDA build.
- **Remote-GPU lane** = anything where the only authoritative pass is `cargo build --release` on the CUDA box, or where the exit gate is a benchmark / e2e test that needs a GPU.
- **Parallel-GPU lane** = independent work the remote box can run **while** the local Mac is busy. Not on the critical path of the active phase.

If a local edit cannot be checked with both `--features no-cuda` AND
`--features metal`, it belongs in remote-GPU lane regardless of how it was
written.

---

## 1 · Phase P0 — `page_size = 16`

### Local (Mac)
- [ ] Edit `infer/src/paged_kv.rs:7,76-87,482-511,549-562,614,760` — add `page_size: usize` field on `TokenKVPool`, rewrite `alloc_tokens` as two-level (new page when `seq_len % page_size == 0`, else append to tail)
- [ ] Edit `infer/src/flashinfer_metadata.rs:125,161` — rewrite incremental update: spill-to-new-page vs bump-`last_page_len`; rewrite the `:161` comment
- [ ] Edit `infer/src/model/qwen3/batch_decode.rs:384` — drop `let page_size = 1;`, read from pool
- [ ] Edit `infer/src/model/qwen35/batch_decode.rs:724` — same
- [ ] Edit `infer/src/model/glm4/batch_decode.rs:338` — same
- [ ] Edit `infer/src/scheduler/cuda/decode.rs:193` — literal `1` → `pool.page_size`
- [ ] `cargo check --no-default-features --features no-cuda` green
- [ ] `cargo check --no-default-features --features metal` green
- [ ] Write the diff, paste into PR draft, hand off to remote GPU

### Remote GPU
- [ ] `cargo build --release` — CUDA compile + FlashInfer link green
- [ ] `cargo test --release` — unit tests pass
- [ ] `cargo test --release --test e2e` — qwen3 greedy parity unchanged
- [ ] `cargo test --release --test e2e_qwen35` — qwen35 greedy parity unchanged
- [ ] `cargo test --release --test greedy_consistency` — multi-config sanity unchanged
- [ ] `scripts/bench_throughput_sweep.py --label page16` — perf snapshot
- [ ] Compare to `--label page1` baseline (collected in §6 below); write `docs/experience/wins/2026-04-13-bench-page16.md`
- [ ] Watch the short-context tail in the sweep — FlashInfer split-KV scheduler can shed parallelism with larger pages at very short contexts. If regression is real, file an error entry and decide whether to ship a short-context fast path

### Exit gate
- e2e + greedy_consistency unchanged
- ≤3% steady-state throughput regression on bench sweep
- Bench markdown committed to `docs/experience/wins/`

---

## 2 · Phase P1 — RadixCache wiring + TierDirectory

### P1 Structural PR (a) — Local (Mac), pure structural
- [ ] Add `infer/src/kv_tier.rs` + `infer/src/kv_tier/{id,directory,tier}.rs` (flat layout)
- [ ] `BlockId::derive` — blake3 reduced to u64 + `parent_hash` chain
- [ ] `BlockHashCtx`, `Tier`, `BlockLocation`, `BlockDescriptor` per project doc §5.1–5.2
- [ ] `TierDirectory` — `DashMap`-backed; `resolve / insert / promote / demote / touch / pin / unpin`
- [ ] Edit `infer/src/prefix_cache.rs` — add `session_id: Option<SessionId>` and `parent_hash: u64` fields on `Node`; derive `Serialize / Deserialize`; recursive parent eviction in `evict()`; change `lookup` return to `(matched_len, Vec<BlockId>)`
- [ ] Add unit tests: serde round-trip, recursive eviction, session-tagged lookup, `BlockId` parent-chain verification, hash collision fallback
- [ ] `infer/src/lib.rs` — `pub mod kv_tier;`
- [ ] `cargo check --no-default-features --features no-cuda` green
- [ ] `cargo check --no-default-features --features metal` green
- [ ] `cargo test --no-default-features --features no-cuda kv_tier prefix_cache` — pure-Rust tests pass
- [ ] Hand off PR (a) for remote GPU smoke build

### P1 Structural PR (a) — Remote GPU
- [ ] `cargo build --release` green (no behavior change expected)
- [ ] `cargo test --release` — entire suite still green; new tests included

### P1 Behavior PR (b) — Local (Mac), edits only
- [ ] Edit `infer/src/scheduler/cuda/core.rs:114-141` — hold `Arc<TieredKvCache>`; remove `cached_prompts: Vec<Vec<u32>>`
- [ ] Edit `infer/src/scheduler/cuda/runtime.rs:75-151` — admission rewrite: radix lookup → block-ref grant → emit prefill chunk for suffix only. Consume `IncomingRequest::session_id` (already plumbed from A2)
- [ ] Edit `infer/src/scheduler/cuda/request.rs` — `ActiveRequest` carries `Vec<BlockId>` accumulator; on finish, commit to directory, release refcounts
- [ ] Edit `infer/src/scheduler/cuda/prefill.rs:14` — read prefix hit length from radix lookup result, drop linear compare
- [ ] Edit `infer/src/server_engine.rs:437-475` — drop the second-source-of-truth `cached_prompt: Vec<u32>`
- [ ] Edit `infer/src/paged_kv.rs` — expose `alloc_slot / free_slot / read_into / write_from` as the T0 physical layer
- [ ] `cargo check --no-default-features --features no-cuda` — Rust types still align across the boundary
- [ ] `cargo check --no-default-features --features metal` — metal path untouched, must remain green

### P1 Behavior PR (b) — Remote GPU
- [ ] `cargo build --release` green
- [ ] `cargo test --release --test e2e` green
- [ ] `cargo test --release --test e2e_qwen35` green
- [ ] `cargo test --release --test greedy_consistency` green
- [ ] **Regression gates** (from `agent-first-architecture.md` §5):
  - `grep -r cached_prompts infer/src/` returns empty
  - `grep -r RadixCache infer/src/scheduler/` returns non-empty
- [ ] Cross-session benchmark on `scripts/bench_agent.py` (built in parallel-GPU §6.3): 2-session alternating trace, prefix hit rate ≥70%
- [ ] Bench markdown in `docs/experience/wins/`

---

## 3 · Phase P2 — T2 host pinned tier + coordinator + auto offload

### P2 Structural PR (a) — Local (Mac)
- [ ] `crates/infer-policy/src/lib.rs` — add `EvictionPolicy` trait, `EvictionCandidate`, `SessionState`, `SessionBiasedLru` default
- [ ] Unit tests for `SessionBiasedLru::score` — pure Rust, fully local
- [ ] Add `infer/src/kv_tier/transport.rs` — `KVTransport` trait + `MemKind` + `TransferOp` + `OpaqueRemoteDesc` per project doc §5.3
- [ ] Add `infer/src/kv_tier/transport/local_cuda.rs` — wraps `cudarc` async copy; mark CUDA-specific items behind `#[cfg(feature = "cuda")]`. Code can be written locally; only test under `cargo build --release` on remote.
- [ ] Add `infer/src/kv_tier/host_pool.rs` — `HostPinnedPool` over `cudaHostAlloc`; allocation-stable base pointer
- [ ] Add `infer/src/kv_tier/coordinator.rs` — tokio task skeleton + in-flight queue + cancel-safe await points; **scheduler does not yet call into it**
- [ ] Add `TieredKvCache::demote / promote` methods that route to coordinator (no-op until behavior PR)
- [ ] `cargo check --no-default-features --features no-cuda` — non-CUDA bits compile
- [ ] `cargo check --no-default-features --features metal` — Metal build untouched

### P2 Structural PR (a) — Remote GPU
- [ ] `cargo build --release` — `cudarc` + `cudaHostAlloc` + dedicated copy stream all link
- [ ] Unit smoke: register a host pinned region, do one async D2H copy, deregister, verify checksums
- [ ] `cargo test --release` — full suite still green (no behavior change yet)

### P2 Behavior PR (b) — Local edits + Remote GPU verify
- [ ] Edit `infer/src/scheduler/cuda/runtime.rs` — eviction hook at admission (`evict_if_needed`) and post-decode (`stamp_keepalive`)
- [ ] Edit `infer/src/scheduler/cuda/core.rs` — pass watermark thresholds into `TieredKvCache`
- [ ] **Diff before delete** — confirm `grep -r 'offload_if_needed\|ensure_on_gpu' infer/src/` returns only `infer/src/model/kv_cache.rs` and its tests
- [ ] **Delete** `infer/src/model/kv_cache.rs:130-168` — `k_host`, `v_host`, `ensure_on_gpu`, `offload_if_needed`. Remove `OFFLOAD_BLOCK_SIZE` constant.
- [ ] `cargo check --features no-cuda` and `--features metal` still green
- [ ] **Remote GPU**: `cargo build --release`, full e2e suite, `greedy_consistency`
- [ ] **Remote GPU**: long-context bench (32k+ cumulative tokens, num_slots=4) that OOMs on main now runs to completion
- [ ] **Remote GPU**: `scripts/bench_throughput_sweep.py --label tier-T2`; ≤3% steady-state regression vs P1 baseline
- [ ] Bench markdown in `docs/experience/wins/`

---

## 4 · Phase P3 — T3 disk tier + session save/load + Metal first contact

P3 has the most local content of any phase. The Metal half is **fully local**.

### P3 Local — disk transport (cross-platform Rust)
- [ ] Add `infer/src/kv_tier/transport/disk.rs`
  - `cfg(target_os = "linux")` → `io_uring` path
  - else → `tokio::fs` fallback
  - `O_DIRECT` where available
- [ ] Allocation-stable region: one large pre-extended file per node, indexed by `(file_id, offset)`
- [ ] Unit tests fully local: write block, read back, hash match — runs on Mac
- [ ] Add `infer/src/http_server/sessions.rs` — `POST /v1/sessions/{id}/save`, `POST /v1/sessions/{id}/load` handlers
- [ ] Edit `infer/src/http_server.rs:422-427` — register routes
- [ ] `crates/infer-agent/src/lib.rs:166-188` — extend `save_to_path / load_from_path` with optional KV side-channel; existing JSON-only path remains
- [ ] `cargo test --no-default-features --features no-cuda kv_tier::transport::disk` — green on Mac

### P3 Local — Metal wired memory bindings
- [ ] Edit `infer/mlx-sys/src/lib.rs` — bindgen + Rust binding for `mlx_metal_set_wired_limit` and `mlx_metal_get_active_memory`
- [ ] Edit `infer/src/metal_kv_pool.rs` — read wired limit at init, cap `max_total_tokens` to avoid the mlx-lm #883 panic (see project doc §8.1)
- [ ] Edit `infer/src/metal_prefix_cache.rs` — disk tier hook through `TieredKvCache` façade
- [ ] `cargo build --release --no-default-features --features metal` — Metal build green on Mac
- [ ] `cargo test --release --no-default-features --features metal` — Metal tests green
- [ ] Long-context Metal smoke test on Mac: 16k+ token session with bounded pool — must NOT panic

### P3 Remote GPU
- [ ] CUDA build with disk transport on; verify io_uring path works on Linux box
- [ ] Restart smoke test: save 30k-token system prompt session, kill process, restart, reload
- [ ] TTFT recovery within 20% of pre-restart warm baseline
- [ ] Bench markdown

### P3 cleanup gate
- [ ] `grep -r session_store infer/` should reference `kv_tier` paths only (the proposed standalone `session_store.rs` never lands)

---

## 5 · Phase P4 — KVFlow-lite reuse-distance + cache-aware routing

### P4 Local (Mac)
- [ ] `crates/infer-policy/src/lib.rs` — `ReuseDistancePolicy` impl; feature flag or config knob to select over `SessionBiasedLru`
- [ ] Unit tests with synthetic turn histories: predicted next-access-time, eviction ordering
- [ ] Edit `infer/src/kv_tier/directory.rs` — per-session turn-arrival ring buffer
- [ ] Edit `infer/src/scheduler/cuda/runtime.rs` — slot selection by radix-subtree overlap, not just emptiness
- [ ] `cargo check` both `--features no-cuda` and `--features metal`

### P4 Remote GPU
- [ ] Cross-session bench, 2-session alternating: prefix hit rate ≥85% (vs ≥70% in P1)
- [ ] KVFlow paper claim reproduction: ≥1.5× target, ≥1.2× minimum to ship as default
- [ ] If reproduction fails: keep `SessionBiasedLru` as default, ship `ReuseDistancePolicy` behind a flag

---

## 6 · Phase P5 — KVTransport trait freeze + NixlTransport stub

### P5 Local (Mac)
- [ ] Add `infer/src/kv_tier/transport/nixl.rs` — `NixlTransport` skeleton; `register / deregister` fully implemented; `put_batch / get_batch` return `todo!("P6")`
- [ ] Edit `infer/Cargo.toml` — add optional `nixl-sys = { version = "0.10", optional = true, features = ["stub-api"] }`; new `rdma-nixl = ["dep:nixl-sys"]` feature
- [ ] `crates/infer-observability` — new `TierTransition` event kind
- [ ] `cargo check --no-default-features --features no-cuda` — default still compiles
- [ ] `cargo check --no-default-features --features no-cuda,rdma-nixl` — stub-api path compiles on Mac
- [ ] Verify trait surface has not changed since P2 — if it has, go back and fix P2

### P5 Remote GPU
- [ ] Optional manual smoke on a CUDA + NIXL native lib box: `cargo check --features cuda,rdma-nixl`. Not a CI gate.

---

## 7 · Parallel-GPU work pool (independent of P0–P5)

Work the remote GPU host can run **while** the local Mac is busy. None of
this is on the tiered-kv critical path; it keeps the GPU productive and
produces data we will need anyway.

### 7.1 — High value, do BEFORE P0

These should already be running by the time P0 edits land:

- [ ] **Baseline collection**: `scripts/bench_throughput_sweep.py --label baseline-main-$(date +%F)` — every model + slot config we ship. Becomes the regression gate from P0 onwards. Save to `docs/experience/wins/2026-04-13-bench-baseline.md`.
- [ ] **`--label page1` baseline**: explicit pre-P0 snapshot under the current `page_size=1` regime. Required for the P0 delta comparison.
- [ ] **Long-context agent baseline**: 32k+ token agent trace, num_slots=4, current main. Numbers we compare against in P2's "must run to completion" gate.
- [ ] **Greedy regression sample**: full `e2e + e2e_qwen35 + greedy_consistency` on current main, capture pass/fail counts as the post-merge baseline.
- [ ] **`scripts/bench_agent.py`** (item C6 from `agent-first-architecture.md`): build the multi-turn tool-calling replayer + input trace under `scripts/data/agent_trace_default.jsonl`. Mostly local Python; GPU validation only. P1 needs this as a scoreboard.

### 7.2 — Medium value, parallel tracks

- [ ] **A3 constrained decoding prototype** (independent of KV tier work): xgrammar-style JSON-schema FSM compiler in `infer/src/constrained/`, sampling-time logit mask in `infer/src/ops/sampling.rs`. Local dev → GPU validation.
- [ ] **A4 speculative decoding scaffolding** (depends on P1; can stage standalone draft model load and CPU-math verify on the GPU side until P1 ships).
- [ ] **Active KV quantization track** — if `docs/projects/kv-quantization-long-context.md` is still active, format-comparison benches keep running on the GPU.

### 7.3 — Background (continuous)

- [ ] **Greedy regression sweep on every main commit**: `cargo test --release --test e2e e2e_qwen35` against a known input set; log timing. Catches silent regressions outside the PR system.
- [ ] **nsys profile collection** for the current decode hot path. Useful as the before-snapshot when P0 lands.
- [ ] **`docs/experience/wins/` benchmark hygiene** — anything missing the environment table per `CLAUDE.md` benchmark rules should be backfilled.

### 7.4 — Lane discipline for parallel work

- Parallel-GPU tasks **never block the critical path**. If a parallel task starts to need the same files as the active phase, pause it, do not race it.
- Every parallel run produces a recorded artifact (test pass count, bench markdown). No invisible runs.
- If the GPU box is genuinely idle (no active phase, no parallel work), pull the next item from §7.2 instead of waiting.

---

## 8 · Concrete next action

1. **Now, remote GPU**: kick off §7.1 baseline collection. Specifically `--label page1` and `--label baseline-main-2026-04-13`. These are **prerequisites** for P0's exit gate — if you don't have a `page1` baseline, you have nothing to compare `page16` against.
2. **Now, local Mac**: start §1 P0 edits. Text-only, ~6 files, all listed with file:line refs.
3. **When P0 edits compile clean** (`cargo check --features no-cuda` and `--features metal`), hand the diff to remote GPU for §1 verification.
4. **While remote GPU runs P0 verification**, immediately start §2 P1 structural PR (a) on the local Mac. P1 (a) is fully local-doable and touches zero P0 files.
5. **Remote GPU after P0 verifies**: hold P0 PR open until baseline benches are recorded; merge P0; immediately start P1 (b) verification cycle once local hands off.

---

## 9 · How this doc is kept honest

- Each task above is a checkbox. Mark it `[x]` when done; do not delete.
- When a phase fully ships, its section header gets a `**Done — see PR #N**` marker. Tasks stay visible as audit trail.
- If a task moves between lanes (e.g. "we thought this was local but cudarc API leaked"), update the lane and add a one-line note. Lane drift is information.
- If a parallel-GPU task starts to look critical-path, promote it into the active phase and remove from §7.

---

## 10 · References

- [`../projects/tiered-kv-cache.md`](../projects/tiered-kv-cache.md) — the project plan this executes
- [`../projects/agent-first-architecture.md`](../projects/agent-first-architecture.md) — A1/B1/B3/C6 source tickets
- [`../../infer/docs/projects/art-grade-architecture-for-long-agent-infer.md`](../../infer/docs/projects/art-grade-architecture-for-long-agent-infer.md) — Phase-1 PR discipline this doc operates under
- `CLAUDE.md` — benchmark rules (immutable history, environment table, raw data not summaries)
