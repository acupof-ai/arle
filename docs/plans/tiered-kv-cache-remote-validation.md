# Tiered KV Cache — remote CUDA validation checklist

**Status**: Active. Consolidates all the local-batch commits (E, H, F,
G, I and their earlier A/B/C/D peers) into a single, mechanical list
of cargo / python commands to run on the remote CUDA host.

**2026-04-15 note**: this checklist was written against the 2026-04-13
P0–P5 phase plan. The project has since re-organised into M0–M5 (see
[`../projects/tiered-kv-cache.md`](../projects/tiered-kv-cache.md) §6 and
[`tiered-kv-cache-tasks.md`](tiered-kv-cache-tasks.md) §0.5 for the
remapping). The **per-commit validation commands in §4 below are
unchanged** — they still match what the P1(a) structural commits did.
The gaps at the end of §4 that were previously labelled "P1 (b)
scheduler swap not yet landed" are now labelled **M1** and are still
the next remote-validation work item when M1 lands. The file paths
referenced below reflect the pre Route-A tree
(`infer/src/paged_kv.rs`, `infer/src/flashinfer_metadata.rs`,
`infer/src/metal_*`, etc.); post Route-A and post the `infer-cuda-kernels`
extraction (2026-04-15 `a4e12f5`), the kernel Rust layer is now under
`crates/infer-cuda-kernels/src/` (`paged_kv.rs`, `flashinfer.rs`,
`graph_pool.rs`, `tensor.rs`, `ffi.rs` + `ffi/*`, `prelude.rs`,
`kv_quant.rs`, `kv_turboquant.rs`, `kv_types.rs`, `turboquant_state.rs`),
and the Metal layer is under `infer/src/backend/metal/*`. Only
`bootstrap.rs` remains in `infer/src/backend/cuda/`. The commands still
work — Cargo resolves them — but when you read a file reference below,
apply the rename mentally.

**For the 2026-04-15 M2b local batch, use**
[`tiered-kv-cache-m2b-remote-acceptance.md`](tiered-kv-cache-m2b-remote-acceptance.md)
**instead.** This doc remains the acceptance record for the older
2026-04-13 structural batch.

This doc is the contract for what "remote validation" means after
2026-04-13's local work. The Mac lane has already run everything that
`cargo check --features no-cuda` and `cargo check --features metal` can
validate; the remote lane is specifically for:

1. The full CUDA build path (`cargo build --release`, default features).
2. `cargo test --release` — files under `backend/cuda/`, `model/*`,
   `scheduler/cuda/*`, `ops/*` are all `#[cfg(feature = "cuda")]`-gated
   and cannot be type-checked on Mac.
3. `cargo test --release --test e2e` / `--test e2e_qwen35` — full
   greedy-decode numerical parity regression against baseline JSONs in
   `infer/test_data/`.
4. `scripts/bench_throughput_sweep.py` and
   `scripts/bench_agent_trace.py` — real throughput and cross-session
   TTFT numbers to feed `docs/experience/wins/`.

Remote validation should run **after** each of the below commits has
been pulled onto the CUDA host. The local lane pushed them directly to
main; no feature branches are involved.

---

## 1 · Environment preflight

Before running any of the checks below, confirm:

- [ ] `git pull origin main` is at the latest local-lane commit (see
  §3 for the commit SHAs).
- [ ] `CUDA_HOME` points at a CUDA 12.x install; `nvcc --version`
  prints.
- [ ] `nvidia-smi` shows at least one GPU.
- [ ] `PEGAINFER_TEST_MODEL_PATH` is set (or the default
  `models/Qwen3-4B` exists) for e2e tests.
- [ ] For optional NIXL validation: `NIXL_PREFIX` points at a NIXL
  install, or set `NIXL_NO_STUBS_FALLBACK=0` (default) to let the
  build fall back to `stubs.cpp`.
- [ ] The working tree is clean (`git status` empty) so benchmark
  output matches the commit it was run against.

---

## 2 · Standing gates (run after every local-lane pull)

These are the baseline checks. If any fails, stop and investigate
before touching the phase-specific items in §4.

```bash
# Full default-feature build (cuda on).
CUDA_HOME=/usr/local/cuda cargo build --release

# Unit tests. Should take ~9s per CLAUDE.md.
cargo test --release

# Clippy with pedantic lints (per infer/Cargo.toml [lints.clippy]).
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

Expected: all green. The local lane verified the no-cuda and metal
lanes already; this is purely the CUDA path.

---

## 3 · Commits covered by this checklist

Ordered chronologically (oldest first). Each commit carries a
"Remote validation" section in its own message — the table below is a
consolidated index.

| # | SHA | Type | Subject |
|---|-----|------|---------|
| 1 | `81f5fb0` | feat(policy)   | EvictionPolicy trait + 4 default impls (A) |
| 2 | `5da8b67` | fix(prefix)    | split must inherit ref_count + evict cascade (B) |
| 3 | `eae8602` | feat(kv_tier)  | module skeleton (single file) (C — superseded by H) |
| 4 | `c531315` | feat(bench)    | bench_agent_trace.py replayer (D) |
| 5 | `ad4996e` | feat(prefix)   | Serialize/Deserialize derive (E) |
| 6 | `b45798f` | refactor(kv_tier) | split into flat layout submodules (H) |
| 7 | `cf60261` | (user commit)  | Qwen3.5 prefill tuning + DiskStore restore (F main body; paired with user's Metal work) |
| 8 | `bee467e` | style(kv_tier) | SessionId::as_str method reference in test |
| 9 | `997d0b7` | feat(kv_tier)  | NIXL transport stub + rdma-nixl feature gate (G) |
| 10 | `8adec3c` | feat(bench)   | /v1/stats probe in bench_agent_trace.py (I) |

**Pre-2026-04-13 context**: the earlier `f47313d` is the enriched
tasks doc itself; `c531315` is the original bench_agent_trace.py;
the four older commits `ad4996e / 5da8b67 / eae8602 / 81f5fb0` were
pushed on the same day. The Qwen3.5 commit (`cf60261`) is the user's
own work — they took the Mac lane's `DiskStore` skeleton out of the
working tree and committed it alongside their prefill path tuning.
Validate that commit as a whole.

---

## 4 · Per-commit validation

### 4.1 EvictionPolicy (`81f5fb0`, task A)

**File**: `infer/src/scheduler/policy.rs` (+341 lines)

```bash
cargo test -p infer --release -- scheduler::policy
# Expected: 19/19 pass (10 legacy admission/chunking + 9 new eviction).

cargo clippy -p infer --release -- -D warnings
# Expected: clean.
```

No CUDA surface. `infer::scheduler::policy` is already backend-agnostic; this
commit does not affect the runtime. Once the P2 behavior PR lands, the
scheduler will pick one of the default `EvictionPolicy` impls.

### 4.2 prefix_cache bug fixes (`5da8b67`, task B)

**File**: `infer/src/prefix_cache.rs` (+207 -30)

```bash
cargo test --release --lib prefix_cache
# Expected: 20/20 pass — 15 legacy + 5 new regression tests:
#   lookup_bumps_every_block_bearing_node_on_path
#   split_node_inherits_ref_count_from_child
#   evict_cascades_through_orphaned_parent_chain
#   evict_cascade_respects_limit_n
#   evict_cascade_respects_ref_count

cargo test --release --test e2e
cargo test --release --test e2e_qwen35
# Expected: no numerical parity regression. The fix changes eviction
# semantics only in cases the existing tests did not previously exercise
# (iterative parent cascade), so greedy decode output must remain
# byte-identical.
```

**Gap** to watch: the prefix cache is still orphaned in P1 (not wired
into the CUDA scheduler yet). These fixes are pre-wiring correctness
improvements. If any `cargo test --release` suite other than
`prefix_cache` changes, that's a red flag — the cache is supposed to
be unused in the data path until A1 lands.

### 4.3 bench_agent_trace.py replayer (`c531315`, task D)

**Files**: `scripts/bench_agent_trace.py`,
`scripts/data/agent_trace_default.jsonl`

```bash
# Prereq: infer server running on the CUDA box.
CUDA_HOME=/usr/local/cuda cargo run -p infer --release -- \
    --model-path models/Qwen3-4B &
SERVER_PID=$!
sleep 10  # wait for model load; adjust as needed

python3 scripts/bench_agent_trace.py \
    --server http://localhost:8000 \
    --label baseline-main-$(date +%F) \
    --out docs/experience/wins/$(date +%F)-bench-agent-trace-baseline.json

kill $SERVER_PID
```

Expected: all 6 sessions in `agent_trace_default.jsonl` succeed. The
aggregate section should show TTFT p50/p99 and ITL p50/p99. The
server-side probe (see 4.10) will also print.

**Baseline** to record: this is the first time this replayer runs
against a real server. Commit the resulting JSON under
`docs/experience/wins/` per the immutable-history rule in `CLAUDE.md`.

### 4.4 prefix_cache serde (`ad4996e`, task E)

**File**: `infer/src/prefix_cache.rs` (+71)

```bash
cargo test --release --lib prefix_cache
# Expected: 22/22 pass — adds radix_cache_serde_roundtrip_preserves_lookups
# and block_id_serde_roundtrip on top of the prior 20.
```

No behavior change; pure derive + round-trip test. Gates the P3 disk
tier session persistence story — once a `DiskStore` + a serde-able
`RadixCache` both exist, the P3 behavior PR can snapshot/restore.

### 4.5 kv_tier skeleton + flat layout (`eae8602` → `b45798f`, tasks C + H)

**Files**: `infer/src/kv_tier.rs`, `infer/src/kv_tier/{id,tier,directory,transport}.rs`

```bash
cargo test --release --lib kv_tier
# Expected: 11/11 pass — 10 directory/tier/id tests + block_id_is_copy_and_ordered.
# (The DiskStore tests under transport/disk.rs add another 8, covered in §4.7.)

cargo clippy --release --lib -- -D warnings
# Expected: clean.
```

`kv_tier` is **always-on** (not cuda-gated). These tests already
passed on Mac; the remote lane is confirming that the full `cargo
build --release` path (which includes cuda modules) does not fight
the new module for any reason.

### 4.6 DiskStore (`cf60261` user commit, task F)

**File**: `infer/src/kv_tier/transport/disk.rs` (+328)

```bash
cargo test --release --lib kv_tier::transport::disk
# Expected: 8/8 pass:
#   rejects_escape_paths
#   round_trips_bytes
#   put_block_and_get_block_roundtrip
#   sequential_put_blocks_advance_file_id
#   get_block_honors_explicit_len
#   get_block_out_of_bounds_errors
#   block_api_rejects_non_disk_location
#   delete_block_is_idempotent
```

Pure `std::fs` + `tempfile`. The remote-lane value is confirming the
tests run identically on Linux (different tmpfs behavior than macOS
`/var/folders/...`). This is the paired deliverable for the user's
Qwen3.5 commit; treat `cf60261` as one atomic unit.

### 4.7 NIXL transport stub (`997d0b7`, task G)

**Files**: `infer/Cargo.toml` (2 features + 1 dep),
`infer/src/kv_tier/transport.rs` (8 lines),
`infer/src/kv_tier/transport/nixl.rs` (+205)

```bash
# Default build must still work — `rdma-nixl` is off by default.
CUDA_HOME=/usr/local/cuda cargo build --release
# Expected: same as §2 baseline.

# Cargo check with stub-api feature — this is the "safe" mode that Mac
# also uses. Should succeed without libnixl installed.
cargo check --features cuda,rdma-nixl
# Expected: compiles, prints a warning if libnixl is absent.

# Run the 4 stub tests. They do NOT call nixl_sys at runtime, only
# reference its types at compile time; linking should succeed because
# Linux has -lstdc++ (the macOS test-link failure does not apply).
cargo test --features cuda,rdma-nixl --release --lib kv_tier::transport::nixl
# Expected: 4/4 pass:
#   stub_transport_reports_its_name
#   stub_put_and_get_return_p5_stub_error
#   stub_register_returns_p5_stub_error
#   stub_poll_returns_ready_error

# Optional: if the CUDA host has a real NIXL install, verify the
# non-stub feature links correctly.
NIXL_PREFIX=/opt/nvidia/nvda_nixl cargo build --features cuda,rdma-nixl-real --release
# Expected: real libnixl.so found, links without error.
```

If `cargo check --features cuda,rdma-nixl` fails on the remote host,
that's a regression from Mac's result and blocks P5 entirely —
file an error entry under `docs/experience/errors/`.

### 4.8 bench_agent_trace `/v1/stats` probe (`8adec3c`, task I)

**File**: `scripts/bench_agent_trace.py` (+158 -4)

```bash
# Rerun §4.3's command; the aggregate section should now include:
#   server /v1/stats:
#     before: requests=0 ... kv_util=0.0% ttft_p50=— ...
#     after:  requests=6 ... kv_util=14.3% ttft_p50=42.5 ...
#     delta:  requests=+6 tokens_out=+1280
#     note:   prefix_hit_rate not exposed by /v1/stats yet; ...
python3 scripts/bench_agent_trace.py \
    --server http://localhost:8000 \
    --label p4-probe-validation-$(date +%F)
```

The "note:" line is expected until a follow-up PR adds prefix hit
counters to `metrics.rs`. If the line is missing, a server-side
counter has been added — update the TODO inside
`fetch_server_stats()` at the same time.

### 4.9 bee467e — trivial style fix, no validation needed

---

## 5 · Consolidated "all green" check

After all per-commit items above pass, run this as the single
end-to-end regression gate before declaring remote validation done:

```bash
# Full tests, including e2e against real weights.
cargo test --release
cargo test --release --test e2e
cargo test --release --test e2e_qwen35
cargo test --release --test greedy_consistency

# Clippy + fmt.
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check

# Throughput sweep for regression detection.
scripts/bench_throughput_sweep.py --label validate-$(date +%F) \
    --out docs/experience/wins/$(date +%F)-bench-throughput-sweep.md

# Agent trace benchmark for the P1 scoreboard.
python3 scripts/bench_agent_trace.py \
    --server http://localhost:8000 \
    --label validate-$(date +%F) \
    --out docs/experience/wins/$(date +%F)-bench-agent-trace.json
```

Commit the benchmark artifacts to `docs/experience/wins/` per the
immutable-history rule. That commit is the "remote validation done"
marker.

---

## 6 · What's explicitly NOT in scope

- **P0 (`page_size = 16`)** — no local-lane PR has touched this yet.
  The enriched tasks doc `docs/plans/tiered-kv-cache-tasks.md` §1
  covers it; remote validation of P0 will get its own checklist entry
  when that PR lands.
- **P1 (b) scheduler swap to radix** — same; the local lane delivered
  B (bug fixes) and C/H (skeleton) as P1 structural work but not the
  behavior PR that rewrites `scheduler/cuda/runtime.rs`.
- **P2 coordinator + host pinned tier** — requires cuda, hasn't been
  attempted locally.
- **Session save/load HTTP routes** — P3 behavior PR, not yet
  touched.
- **KVTransport trait impl over DiskStore** — the skeleton exists
  but the trait impl for disk is deferred to the same P3 behavior
  PR.

If the remote lane hits a blocker during any of §4's checks, the
right response is almost always "update
`docs/plans/tiered-kv-cache-tasks.md` §N.3 Course corrections, then
file a new local-lane task," not "patch on the CUDA host directly."
The Mac lane is the source of truth for everything that compiles
there; the CUDA host is the source of truth for everything that
doesn't.

---

## 7 · Sanity grep — what the local lane actually touched

Run this on the CUDA host after `git pull` to confirm the tree is in
the expected state. Any mismatch with this list points at a bad pull
or a dirty working tree.

```bash
# Should exist:
test -f infer/src/scheduler/policy.rs             # A (EvictionPolicy)
test -f infer/src/prefix_cache.rs                  # B, E
test -f infer/src/kv_tier.rs                       # C, H (entrypoint)
test -f infer/src/kv_tier/id.rs                    # H
test -f infer/src/kv_tier/tier.rs                  # H
test -f infer/src/kv_tier/directory.rs             # H
test -f infer/src/kv_tier/transport.rs             # H
test -f infer/src/kv_tier/transport/disk.rs        # F (via cf60261)
test -f infer/src/kv_tier/transport/nixl.rs        # G
test -f scripts/bench_agent_trace.py               # D, I
test -f scripts/data/agent_trace_default.jsonl     # D

# Feature flags that should be present in infer/Cargo.toml:
grep -q 'rdma-nixl = '      infer/Cargo.toml
grep -q 'rdma-nixl-real = ' infer/Cargo.toml
grep -q 'nixl-sys.*stub-api' infer/Cargo.toml

# Unit test targets:
grep -q 'EvictionPolicy'          infer/src/scheduler/policy.rs
grep -q 'radix_cache_serde_roundtrip_preserves_lookups' infer/src/prefix_cache.rs
grep -q 'split_node_inherits_ref_count_from_child'      infer/src/prefix_cache.rs
grep -q 'evict_cascades_through_orphaned_parent_chain'  infer/src/prefix_cache.rs
grep -q 'put_block_and_get_block_roundtrip'             infer/src/kv_tier/transport/disk.rs
grep -q 'stub_transport_reports_its_name'               infer/src/kv_tier/transport/nixl.rs
```

All should exit 0. If any fail, `git status` and re-pull before
running §4.

---

## 8 · References

- `docs/plans/tiered-kv-cache-tasks.md` — the lane split and the
  per-phase research summaries.
- `docs/projects/tiered-kv-cache.md` — the design doc that all of
  this implements.
- `docs/projects/agent-first-architecture.md` — A1 / B1 / B3 source
  tickets; the local-lane commits above are their implementation.
- `CLAUDE.md` — the benchmark rules and bench hygiene
  (`scripts/bench_throughput_sweep.py --label`, immutable history,
  environment tables, raw data).
