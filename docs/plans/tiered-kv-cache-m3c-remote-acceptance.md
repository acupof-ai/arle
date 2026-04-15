# Tiered KV Cache M3c — remote CUDA acceptance

**Status**: Active. This is the acceptance contract for the 2026-04-15
local M3c cleanup batch.

**Scope under test**:
- The legacy contiguous CPU KV offload path is deleted from production code:
  no `OFFLOAD_BLOCK_SIZE`, no `k_host/v_host`, no
  `prefetch_kv_to_gpu/offload_kv_if_needed`, no `ensure_on_gpu`.
- CUDA scheduler prefix reuse now assumes resident-only contiguous KV before
  paged migration.
- Single-request engine / CLI keep `set_max_gpu_kv` only as a compatibility
  no-op warning.
- `tests/test_kv_cache.py` now mirrors the resident-only metadata and
  `TokenKVPool` budget semantics that still exist.

**Explicit non-scope**:
- No real T1 runtime staging completion yet: the current tree may already
  include the M3b contract/runtime-wire batch (`lookup_or_stage`,
  keepalive stamping, watermark rewiring), but staged bytes still do not
  complete back onto GPU on this lane.
- No real host-pinned promotion path yet.
- No disk/session persistence yet.

This doc assumes the CUDA host has already run the structural gates in
[`tiered-kv-cache-m0.3-m3a-remote-acceptance.md`](tiered-kv-cache-m0.3-m3a-remote-acceptance.md)
and the contract/state-machine gate in
[`tiered-kv-cache-m3b-remote-acceptance.md`](tiered-kv-cache-m3b-remote-acceptance.md).

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
rg -n "OFFLOAD_BLOCK_SIZE|prefetch_kv_to_gpu|offload_kv_if_needed|prefetch_to_gpu|ensure_on_gpu|k_host|v_host" \
  infer/src/model.rs \
  infer/src/model/generation_state.rs \
  infer/src/model/kv_cache.rs \
  infer/src/model/qwen3/forward.rs \
  infer/src/model/qwen35/forward.rs \
  infer/src/model/glm4/forward.rs \
  infer/src/scheduler/cuda/prefill.rs \
  infer/src/scheduler/cuda/runtime.rs \
  infer/src/server_engine.rs
```

Expected: **no output**.

```bash
rg -n "set_max_gpu_kv" infer/src/server_engine.rs crates/infer-cli/src/{args,lib}.rs
```

Expected: matches only for the compatibility shim / warning path.

```bash
python -m pytest tests/test_kv_cache.py -q
```

Expected: all tests pass; the file no longer models CPU offload behavior.

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
- [ ] No new CUDA-only linker/runtime failure is introduced by the M3c
      cleanup.
- [ ] Golden outputs remain unchanged.

---

## 4 · Long-session regression gate

Run a long-session workload that previously depended on the old offload path
being dormant rather than correct. Reuse the same host / same model / same
flags when comparing to earlier M2b/M3a baselines.

Suggested server launch:

```bash
CUDA_HOME=/usr/local/cuda cargo run -p infer --release -- \
  --model-path models/Qwen3-4B
```

Then, in another shell:

```bash
python3 scripts/bench_agent_trace.py \
  --server http://localhost:8000 \
  --label tiered-kv-m3c-remote \
  --out docs/experience/wins/2026-04-15-bench-tiered-kv-m3c-remote.json
```

Acceptance:
- [ ] The run completes without CUDA faults or prefix-reuse regressions.
- [ ] Repeated-session TTFT does not regress materially versus the accepted
      M2b/M3a host baseline.
- [ ] Server logs do not show stale-prefix replay failures caused by the
      legacy offload removal.

---

## 5 · Sign-off checklist

M3c is accepted when all of the below are true:

- [ ] Static sanity checks passed.
- [ ] Full build/test gate passed on CUDA.
- [ ] Long-session regression gate completed without runtime faults.
- [ ] A win note was written under `docs/experience/wins/` with:
      environment, commands, raw outputs or linked artifacts, and explicit
      comparison against the pre-M3c accepted host baseline.

After sign-off, update:
- `docs/projects/tiered-kv-cache.md` — M3c status from
  "local cleanup shipped; remote validation pending" to "accepted".
- `docs/plans/tiered-kv-cache-tasks.md` — mark the remote checklist done.
- `docs/index.md` — add the new win note if it is not already listed.
