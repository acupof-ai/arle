# Tiered KV Cache Tier A/B/C — remote CUDA acceptance

**Status**: Active. This is the combined acceptance contract for the
2026-04-16 local Tier A/B/C follow-on batch on top of the already accepted
M2b + M0.3/M3a + M3b + M3c stack.

**Scope under test**:
- Tier A (`d3d1e46`): the CUDA scheduler now owns a live coordinator thread,
  passes a real `CoordinatorHandle` as `StagePlanner`, parks staged requests
  in `stage_waiting`, and re-admits them when `StagingCompleted` arrives.
- Tier B (`e0f69f9`): publish computes `BlockFingerprint` per block and routes
  inserts through `RadixCache::insert_with_fingerprints(...)`; DiskStore round
  trips bytes plus fingerprint locally.
- Tier C (`9b01c2a`): `RadixCache` keeps a private `block_index` for O(1)
  block lookup, and the prefix-cache watermarks / keepalive knobs now live on
  `SchedulerConfig` with validation.

**Explicit non-scope**:
- No real async `cudaMemcpyAsync` completion yet. The current local Tier A
  transport still emits `StagingQueued` plus a synchronous `StagingCompleted`
  echo.
- No DiskStore wiring on the coordinator `Stage` command path yet.
- No new disk/session persistence acceptance beyond the local fingerprint
  round-trip test.

This doc assumes the CUDA host has already run the earlier acceptance docs:
[`tiered-kv-cache-m2b-remote-acceptance.md`](tiered-kv-cache-m2b-remote-acceptance.md),
[`tiered-kv-cache-m0.3-m3a-remote-acceptance.md`](tiered-kv-cache-m0.3-m3a-remote-acceptance.md),
[`tiered-kv-cache-m3b-remote-acceptance.md`](tiered-kv-cache-m3b-remote-acceptance.md),
and [`tiered-kv-cache-m3c-remote-acceptance.md`](tiered-kv-cache-m3c-remote-acceptance.md).

---

## 1 · Preflight

- [ ] `git status --short` is clean or only contains the intended stacked diff.
- [ ] `git rev-parse --abbrev-ref HEAD` points at the branch to validate.
- [ ] `nvidia-smi` shows the target GPU.
- [ ] `CUDA_HOME=/usr/local/cuda` (or the correct local CUDA path) exists.
- [ ] `PEGAINFER_TEST_MODEL_PATH` points at a valid test model, or
      `models/Qwen3-4B` exists locally.

---

## 2 · Static sanity checks

Run these before any long build/test job:

```bash
rg -n "coordinator_handle|stage_waiting|page_lifecycle" \
  infer/src/scheduler/cuda infer/src/kv_tier
```

Expected: scheduler/runtime matches for the live coordinator handle,
ticket-wait map, and page-lifecycle ownership.

```bash
rg -n "insert_with_fingerprints|BlockFingerprint::compute_from_tokens" \
  infer/src/prefix_cache.rs infer/src/scheduler/cuda/core.rs infer/src/types.rs
```

Expected: publish path computes fingerprints and routes through the new
canonical insert API.

```bash
rg -n "block_index|rebuild_block_index" infer/src/prefix_cache.rs
```

Expected: one private index field plus the post-serde rebuild helper.

```bash
rg -n "struct SchedulerConfig|prefix_cache_high_water|prefix_cache_low_water|prefix_cache_retain_hard_cap|prefix_cache_keepalive_ticks|stage_wait_keepalive_ticks" \
  infer/src/scheduler/types.rs infer/src/scheduler/cuda/core.rs
```

Expected: the five knobs exist on `SchedulerConfig`, are validated there, and
the runtime reads the config fields rather than deleted module consts.

---

## 3 · Build and test gates

```bash
CUDA_HOME=/usr/local/cuda cargo build --release
cargo test --release
PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e
PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e_qwen35
cargo test --release --test greedy_consistency
cargo clippy --workspace -- -D warnings
```

Acceptance:
- [ ] All six commands pass.
- [ ] No new CUDA-only linker/runtime failure is introduced by the Tier A/B/C
      follow-on batch.
- [ ] Golden outputs remain unchanged.

---

## 4 · Long-session regression gate

Re-use the same host / model / flags as the accepted M3c remote baseline.
The goal here is not to prove real async staging yet; it is to confirm that
the Tier A stub-completion path and Tier B/C metadata changes do not regress
the already accepted M3c behavior envelope.

Suggested server launch:

```bash
CUDA_HOME=/usr/local/cuda cargo run -p infer --release -- \
  --model-path models/Qwen3-4B
```

Then, in another shell:

```bash
python3 scripts/bench_agent_trace.py \
  --server http://localhost:8000 \
  --label tiered-kv-tier-abc-remote \
  --out docs/experience/wins/2026-04-16-bench-tiered-kv-tier-abc-remote.json
```

Acceptance:
- [ ] The run completes without CUDA faults, deadlocks, or stuck staged
      requests.
- [ ] Repeated-session TTFT stays within noise of the accepted
      `docs/experience/wins/2026-04-15-tiered-kv-m3c-remote.md` baseline.
- [ ] Server logs show the Tier A stub path behaving as expected:
      staged hits queue, complete, and re-enter admission without regressing
      the pre-existing M3c flow.

---

## 5 · Sign-off checklist

Tier A/B/C is accepted when all of the below are true:

- [ ] Static sanity checks passed.
- [ ] Full build/test gate passed on CUDA.
- [ ] Long-session regression gate completed without runtime faults.
- [ ] A win note was written under `docs/experience/wins/` with:
      environment, commands, raw outputs or linked artifacts, and explicit
      comparison against the accepted M3c remote baseline.

After sign-off, update:
- `docs/projects/tiered-kv-cache.md` — mark the Tier A/B/C follow-on as
  CUDA-accepted rather than local-only.
- `docs/plans/tiered-kv-cache-tasks.md` — mark this checklist done.
- `docs/index.md` — add the new win note if it is not already listed.
