# Tiered KV Cache M2b — remote CUDA acceptance

**Status**: Active. This is the acceptance contract for the 2026-04-15
local M2b batch.

**Scope under test**:
- CUDA scheduler admission now uses radix-driven reusable-prefix selection.
- Scheduler-owned `cached_prompts: Vec<Vec<u32>>` is deleted.
- Prefix reuse prefetches CPU-offloaded contiguous KV before any
  `truncate_to()` / `migrate_kv_range_to_paged()` path reads it.
- Pool allocation retry (`alloc_pool_tokens_with_retry`) can force one
  synchronous prefix-cache eviction on OOM.
- Retain hard cap (`0.90`) and radix tombstone GC are active.
- **Explicit non-scope**: no cross-slot paged-pool aliasing. Reuse is safe
  same-slot resurrection only.

This doc turns M2b from "local implementation landed" into "accepted on a
CUDA host". If a step fails, stop and file the failure under
`docs/experience/errors/`.

---

## 1 · Preflight

- [ ] `git status --short` is clean or only contains the intended M2b diff.
- [ ] `git rev-parse --abbrev-ref HEAD` points at the branch to validate.
- [ ] `nvidia-smi` shows the target GPU.
- [ ] `CUDA_HOME=/usr/local/cuda` (or the correct local CUDA path) exists.
- [ ] `PEGAINFER_TEST_MODEL_PATH` points at a valid test model, or
      `models/Qwen3-4B` exists locally.
- [ ] Any previous server on `:8000` is stopped.

---

## 2 · Static sanity checks

Run these before any long build/test job:

```bash
rg -n "cached_prompts: Vec<Vec<u32>>|best_prefix_slot_for_cached_prompts" infer/src/scheduler/cuda
```

Expected: **no output**.

```bash
rg -n "reusable_prefix_len|reusable_cached_prompt_len|block_owner_slots|slot_materialized_prompt_lens|alloc_pool_tokens_with_retry|PREFIX_CACHE_RETAIN_HARD_CAP" infer/src/scheduler/cuda
```

Expected: matches in `request.rs`, `runtime.rs`, `core.rs`, and `prefill.rs`.

```bash
rg -n "prefetch_kv_to_gpu" infer/src/model.rs infer/src/model/qwen3/forward.rs infer/src/model/qwen35/forward.rs infer/src/model/glm4/forward.rs
```

Expected: one trait method plus one implementation in each CUDA model state.

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
- [ ] No new CUDA-only linker/runtime failure is introduced by the M2b diff.
- [ ] Golden outputs remain unchanged.

---

## 4 · Agent trace benchmark gate

Start the server on the CUDA host:

```bash
CUDA_HOME=/usr/local/cuda cargo run -p infer --release -- \
  --model-path models/Qwen3-4B
```

In another shell, run:

```bash
python3 scripts/bench_agent_trace.py \
  --server http://localhost:8000 \
  --label tiered-kv-m2b-remote \
  --out docs/experience/wins/2026-04-15-bench-tiered-kv-m2b-remote.json
```

Compare against the M2a baseline on the **same host / same model / same
flags**. If no baseline exists yet on that host, first run the same benchmark
against commit `4402ab0` and save it before accepting M2b.

Acceptance:
- [ ] Repeated-session TTFT is lower than the M2a baseline.
- [ ] No obvious regression in steady-state ITL.
- [ ] Server logs show reusable-prefix admissions rather than only cold
      prefill on the repeated-prefix turns.

Record the result in a new win note under `docs/experience/wins/`.

---

## 5 · Shared-prefix stress gate

With the same server still running, launch a burst of requests that all share
one long prefix:

```bash
python3 - <<'PY'
import asyncio
import httpx

PREFIX = (
    "System: you are a precise coding assistant. "
    + "Reuse this long prefix. " * 512
)

async def one(i: int):
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(
            "http://localhost:8000/v1/completions",
            json={
                "model": "Qwen3-4B",
                "prompt": PREFIX + f"User question #{i}: explain one scheduler invariant.",
                "max_tokens": 32,
                "temperature": 0,
            },
        )
        return i, resp.status_code, resp.text[:120]

async def main():
    results = await asyncio.gather(*[one(i) for i in range(100)])
    bad = [r for r in results if r[1] != 200]
    print(f"total={len(results)} bad={len(bad)}")
    for row in bad[:10]:
        print(row)

asyncio.run(main())
PY
```

Acceptance:
- [ ] `bad=0`.
- [ ] Server logs do **not** show pool-allocation failure loops.
- [ ] Retain hard cap causes skip/fail-open behavior under pressure rather
      than admission starvation.

---

## 6 · Sign-off checklist

M2b is accepted when all of the below are true:

- [ ] Static sanity checks passed.
- [ ] Full build/test gate passed on CUDA.
- [ ] Agent-trace benchmark beat or matched the M2a baseline on TTFT.
- [ ] Shared-prefix stress test produced no allocation failures.
- [ ] A win note was written under `docs/experience/wins/` with:
      environment, commands, raw outputs or linked artifacts, and the M2a
      comparison.

After sign-off, update:
- `docs/projects/tiered-kv-cache.md` — M2b status from "local impl done;
  remote validation pending" to "accepted".
- `docs/plans/tiered-kv-cache-tasks.md` — mark the remote checklist done.
- `docs/index.md` — add the new win note if it is not already listed.
