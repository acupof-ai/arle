# Tiered KV Cache M3b (contract tranche) — remote CUDA acceptance

**Status**: Active. This is the acceptance contract for the 2026-04-15
local M3b contract/state-machine batch.

**Scope under test**:
- `RadixCache::lookup_or_stage(...)` exists and classifies GPU / host /
  disk / tombstone hits through `HitKind`.
- `LookupOutcome`, `LookupHeuristics`, `StageTicket`, and `StagePlanner`
  are exported at `crate::kv_tier::*`.
- `CoordinatorHandle` can emit ticketed staging commands/events.
- The pure `Free | Resident | Demoting` page-lifecycle state machine is
  in-tree and covered by tests.
- `RadixCache::evict_with_policy(...)` remains the live eviction path for
  scheduler cleanup/allocation.

**Explicit non-scope**:
- No live scheduler/runtime consumer of `lookup_or_stage` yet.
- No watermark retune in CUDA scheduler yet.
- No real T1↔T0 promotion path yet.
- No deletion of legacy `model/kv_cache.rs` CPU offload yet.

This doc turns the M3b local batch from "contract landed" into "accepted on
a CUDA host". If a step fails, stop and record the failure under
`docs/experience/errors/`.

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
rg -n "lookup_or_stage|set_block_location|set_block_byte_len" infer/src/prefix_cache.rs
```

Expected: matches for the new lookup contract helpers.

```bash
rg -n "HitKind|LookupOutcome|LookupHeuristics|StageTicket|StagePlanner|PageLifecycleState" infer/src/kv_tier infer/src/kv_tier.rs
```

Expected: matches in `kv_tier/lookup.rs`, `kv_tier/coordinator.rs`, and
`kv_tier.rs`.

```bash
rg -n "lookup_or_stage|StageTicket|PageLifecycleState" infer/src/scheduler/cuda
```

Expected: **no output**. This batch should not have silently wired live
CUDA runtime behavior yet.

```bash
rg -n "evict_with_policy" infer/src/prefix_cache.rs infer/src/scheduler/cuda/core.rs
```

Expected: matches in both files; cleanup/allocation still flow through the
policy-scored path.

---

## 3 · Build and test gates

```bash
CUDA_HOME=/usr/local/cuda cargo build --release
cargo test --release
PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e
PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e_qwen35
cargo test --release --test greedy_consistency
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

Acceptance:
- [ ] All commands pass.
- [ ] No new CUDA-only linker/runtime failure is introduced by the M3b
      contract tranche.
- [ ] Golden outputs remain unchanged.

---

## 4 · Focused M3b smoke

The new contract/state-machine code should still be explicitly exercised on
the CUDA host:

```bash
cargo test -p infer --release prefix_cache
cargo test -p infer --release kv_tier
```

Acceptance:
- [ ] `prefix_cache` tests pass, including the new `lookup_or_stage`
      classifications.
- [ ] `kv_tier` tests pass, including `StageTicket` staging events and
      `PageLifecycleState` transitions.
- [ ] `LocalCudaTransport` structural tests still pass on the CUDA host.

Optional explicit grep smoke:

```bash
rg -n "GPU required: LocalCudaTransport poll is a structural stub" \
  infer/src/kv_tier/transport/local_cuda.rs
```

Expected: present. Real `cudaMemcpyAsync` behavior is still a later M3
runtime patch.

---

## 5 · Sign-off checklist

M3b contract/state-machine tranche is accepted when all of the below are true:

- [ ] Static sanity checks passed.
- [ ] Full build/test gate passed on CUDA.
- [ ] Focused `prefix_cache` / `kv_tier` smoke passed on CUDA.
- [ ] A win note was written under `docs/experience/wins/` with:
      environment, commands, raw outputs or linked artifacts, and an
      explicit note that runtime staging/watermark rewiring are still
      pending.

After sign-off, update:
- `docs/projects/tiered-kv-cache.md` — M3b status from
  "local contract/state-machine + policy convergence landed; runtime wire
  and remote validation pending" to "CUDA-accepted contract tranche;
  runtime wire still pending".
- `docs/plans/tiered-kv-cache-tasks.md` — mark this remote checklist done.
- `docs/index.md` — add the new win note if it is not already listed.
