# Tiered KV Cache M4 — remote CUDA acceptance

**Status**: Active. This is the acceptance contract for the
2026-04-16 local M4 batch (A/B/C/D), stacked on top of the
already-accepted M2b + M0.3 + M3a + M3b + M3c + Tier A/B/C local
runtime batches.

**Scope under test**:

- **M4a** (`66d38ad`): `BlockFingerprint::compute(ctx, tokens)` is
  a real BLAKE3 hash over a domain-tagged input chain
  (`model_fingerprint`, `kv_format_tag`, `parent`, `tokens`).
  `Scheduler<M>` owns a `model_fingerprint: Vec<u8>` derived at
  construction via `blake3::hash(model_id.as_bytes())`. Publish
  chains parent fingerprints block-by-block.
- **M4b** (`c7cc0d6`): `DiskStore` stores blocks as content-
  addressable files named by the 16-byte fingerprint hex +
  `.kv`. Files start with a postcard-encoded
  `DiskBlockHeader { magic: b"PEGAKV01", version: 1, fingerprint,
  kv_format_tag, payload_len }` followed by raw payload bytes.
  `put_block` / `get_block` / `delete_block` take
  `DiskBlockLocation { path, payload_len, fingerprint }`.
- **M4c** (`7b72d02`): `RadixCache::reconcile(known: &HashMap<
  BlockFingerprint, BlockId>) -> ReconcileReport`. Post-
  deserialization pass that remaps block_ids against a fresh
  pool, tombstones unknown fingerprints, drops orphan nodes, and
  rebuilds the O(1) `block_index`. Runtime-only fields
  (`ref_count`, `last_access`, `soft_pin_until`, `clock`) are
  `#[serde(default, skip)]`.
- **M4d** (`c87c68b`): `infer/src/http_server/sessions.rs` —
  pure-Rust `save_session(session_id, kv_format_tag, radix, disk,
  payload_for, fingerprints)` + `load_session(snapshot, disk,
  allocate_block_id) -> LoadedSession { radix, kv_payloads,
  report }`. **No HTTP routes** in this batch — handlers come in
  a follow-up.

**Explicit non-scope**:

- No axum `Router::route` entry for session save/load yet. The
  module is ready to wrap but deliberately unwrapped.
- No real T1→T2 spill path through the coordinator. `DiskStore`
  is callable directly but no `CoordinatorCommand::DemoteToDisk`
  exists yet.
- No MLX wired-memory bindings on the Metal side. Metal backend
  stays T0-only until a separate batch adds those bindings.
- No BLAKE3 weight-checksum upgrade for `model_fingerprint`. The
  current `blake3::hash(model_id.as_bytes())` is per-engine
  stable; real content checksumming is M5-era work.

This doc assumes the CUDA host has already run the earlier
acceptance docs:
[`tiered-kv-cache-m2b-remote-acceptance.md`](tiered-kv-cache-m2b-remote-acceptance.md),
[`tiered-kv-cache-m0.3-m3a-remote-acceptance.md`](tiered-kv-cache-m0.3-m3a-remote-acceptance.md),
[`tiered-kv-cache-m3b-remote-acceptance.md`](tiered-kv-cache-m3b-remote-acceptance.md),
[`tiered-kv-cache-m3c-remote-acceptance.md`](tiered-kv-cache-m3c-remote-acceptance.md),
and
[`tiered-kv-cache-tier-abc-remote-acceptance.md`](tiered-kv-cache-tier-abc-remote-acceptance.md).

---

## 1 · Preflight

- Order reminder for reviewers: M2b → M0.3/M3a → M3b contract →
  M3b runtime wire → M3c cleanup → Tier A/B/C runtime promotion →
  **M4 a/b/c/d local** (this doc).
- [ ] `git status --short` is clean or only contains the intended
      stacked diff.
- [ ] `git rev-parse --abbrev-ref HEAD` points at the branch to
      validate.
- [ ] `nvidia-smi` shows the target GPU.
- [ ] `CUDA_HOME=/usr/local/cuda` (or the correct local CUDA
      path) exists.
- [ ] `INFER_TEST_MODEL_PATH` points at a valid test model,
      or `models/Qwen3-4B` exists locally.

---

## 2 · Static sanity checks

Run these before any long build/test job:

```bash
rg -n "BlockFingerprint::compute|KvContentContext" \
  infer/src/types.rs infer/src/scheduler/cuda/core.rs
```

Expected: `compute` definition in `types.rs`, one call site inside
`publish_to_prefix_cache` in `core.rs` with the full parent-
chained loop.

```bash
rg -n "stable_tag" crates/infer-cuda-kernels/src/kv_types.rs \
  infer/src/scheduler/cuda/core.rs
```

Expected: `stable_tag()` definition on `KVFormat` plus its `#[test]
stable_tags_are_fixed`, and one caller inside
`publish_to_prefix_cache`.

```bash
rg -n "DiskBlockHeader|DISK_BLOCK_MAGIC|b\"PEGAKV01\"" \
  infer/src/kv_tier/transport/disk.rs
```

Expected: the header struct, the magic constant, and the
`DiskBlockHeader::decode` path.

```bash
rg -n "fn reconcile|ReconcileReport" infer/src/prefix_cache.rs
```

Expected: struct definition + method definition. **No**
`rebuild_from_fingerprints` — it was deleted in M4c.

```bash
rg -n "save_session|load_session|SessionSnapshot" \
  infer/src/http_server/sessions.rs infer/src/http_server.rs
```

Expected: module declaration in `http_server.rs` + the two
functions + the snapshot types in `sessions.rs`.

```bash
rg -n "rebuild_from_fingerprints" infer/ docs/
```

Expected: **empty**. If anything still references the deleted
helper, that's a doc drift to fix before sign-off.

```bash
rg -n "compute_from_tokens" infer/ docs/
```

Expected: only test code and the `#[doc(hidden)]` shim definition
itself. Any non-test production call site is a latent persistence
bug because the shim uses empty context — flag and fix.

---

## 3 · Build and test gates

```bash
CUDA_HOME=/usr/local/cuda cargo build --release
cargo test --release
INFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e
INFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e_qwen35
cargo test --release --test greedy_consistency
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

Acceptance:

- [ ] All commands pass.
- [ ] No new CUDA-only linker/runtime failure introduced by the
      M4 a/b/c/d batch.
- [ ] Golden outputs unchanged.
- [ ] `infer::http_server::sessions::tests` run in the default
      `cargo test --release` pass.

Targeted test passes (fast subset, useful for iteration):

```bash
cargo test -p infer --release --no-default-features --features no-cuda fingerprint
cargo test -p infer --release --no-default-features --features no-cuda prefix_cache
cargo test -p infer --release --no-default-features --features no-cuda disk
cargo test -p infer --release --no-default-features --features no-cuda sessions
```

Expected counts as of the 2026-04-16 local batch: 9 fingerprint /
44 prefix_cache / 9 disk / 3 sessions. Totals should match the
`cargo test -p infer --release --no-default-features --features no-cuda`
result of 276 passing tests (up from 267 pre-M4).

---

## 4 · Long-session regression gate

Re-use the same host / model / flags as the accepted Tier A/B/C
remote baseline
(`docs/experience/wins/2026-04-16-tiered-kv-tier-abc-remote.md`
once that checklist has been signed off — otherwise fall back to
the M3c baseline at
`docs/experience/wins/2026-04-15-tiered-kv-m3c-remote.md`).

The M4 local batch does NOT introduce new scheduler behaviors
(staging, publish path, admission) beyond what Tier A/B/C already
shipped — BLAKE3 replaces a hash, postcard replaces raw bytes,
reconcile + sessions are new but not on the hot path. The goal of
the regression gate is to confirm the hash / disk / reconcile
changes do not leak latency into publish or admission under a
real decode workload.

Suggested server launch:

```bash
CUDA_HOME=/usr/local/cuda cargo run -p infer --release -- \
  --model-path models/Qwen3-4B
```

Then, in another shell:

```bash
# Replaces the deprecated bench_throughput_sweep.py — use
# guidellm per infer/CLAUDE.md §Benchmarks.
scripts/bench_guidellm.sh tiered-kv-m4-remote
```

Acceptance:

- [ ] Run completes without CUDA faults, deadlocks, or stuck
      requests.
- [ ] Repeated-session TTFT stays within noise of the accepted
      Tier A/B/C remote baseline.
- [ ] Publish-path CPU time is not materially higher (fingerprint
      compute on BLAKE3 over per-block tokens should be
      micro-seconds, but confirm).
- [ ] Server logs do not show fingerprint mismatch warnings or
      unexpected reconcile events (reconcile is not called at
      runtime in this batch — any `reconcile` log is a surprise).

---

## 5 · Session round-trip smoke (optional on L4)

Exercises the new M4d save/load path end-to-end on the CUDA host.
Optional because the unit tests already cover the Rust surface;
running it on CUDA confirms the postcard wire format survives a
real `DiskStore` root directory.

```bash
cargo test -p infer --release --features cuda sessions::tests::save_then_load_round_trips_radix_and_payloads
cargo test -p infer --release --features cuda sessions::tests::save_skips_blocks_with_no_payload
cargo test -p infer --release --features cuda sessions::tests::load_errors_on_tampered_disk_payload
```

Acceptance:

- [ ] All three tests pass under the `cuda` feature build.
- [ ] `/tmp/<tempdir>/` directories contain files named by
      lowercase 16-byte fingerprint hex + `.kv`.

---

## 6 · Sign-off checklist

M4 is accepted when all of the below are true:

- [ ] Static sanity checks passed.
- [ ] Full build/test gate passed on CUDA.
- [ ] Long-session regression gate completed without runtime
      faults.
- [ ] A win note was written under `docs/experience/wins/` with:
      environment, commands, raw outputs or linked artifacts, and
      explicit comparison against the accepted Tier A/B/C remote
      baseline.

After sign-off, update:

- `docs/projects/tiered-kv-cache.md` — mark the M4 local batch as
  CUDA-accepted rather than local-only, and move the "first Metal
  contact" bullet point out of M4 scope into a separate deferred
  entry (Metal MLX wired-memory bindings were intentionally cut
  from this batch).
- `docs/plans/tiered-kv-cache-tasks.md` — mark this checklist
  done.
- `docs/index.md` — add the new win note if not already listed.
