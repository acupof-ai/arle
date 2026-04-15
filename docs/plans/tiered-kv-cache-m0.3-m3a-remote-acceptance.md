# Tiered KV Cache M0.3 + M3a — remote CUDA acceptance

**Status**: Active. This is the acceptance contract for the 2026-04-15
local M0.3 + M3a batch.

**Scope under test**:
- BF16 paged-KV now defaults to `page_size = 16`; INT8 / FP8 / TurboQuant
  intentionally remain at `page_size = 1`.
- `TokenKVPool` is page-aware (`free_pages`, `page_indices`, `seq_lens`,
  `max_total_pages`) instead of token-indexed.
- BF16 range migration uses the new HND-aware
  `kv_cache_to_paged_range_hnd_cuda` kernel.
- FlashInfer decode metadata and CUDA decode planning now read runtime
  `page_size`.
- `HostPinnedPool`, `LocalCudaTransport`, `Coordinator`, and the first
  tier-aware `RadixNode` metadata fields are in-tree as the M3a structural
  skeleton.

**Explicit non-scope**:
- No M3 behavior yet: no scheduler demote/promote path, no watermark trigger,
  no live `lookup_or_stage`, no real T1↔T0 promotion.
- No quantized `page_size > 1` path yet; quantized formats stay token-granular.
- No cross-slot page aliasing; that remains outside M2b/M3a.

This doc assumes the CUDA host has already run the scheduler-side M2b gate in
[`tiered-kv-cache-m2b-remote-acceptance.md`](tiered-kv-cache-m2b-remote-acceptance.md).

---

## 1 · Preflight

- [ ] `git status --short` is clean or only contains the intended stacked diff.
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
rg -n "default_page_size|page_size = 16|max_total_pages|free_pages|page_indices|seq_lens" \
  crates/infer-cuda-kernels/src/{kv_types,paged_kv,flashinfer}.rs
```

Expected:
- BF16 resolves to `page_size = 16`
- page-aware pool fields exist
- FlashInfer metadata references runtime `page_size`

```bash
rg -n "kv_cache_to_paged_range_hnd_cuda|kv_cache_to_paged_range_hnd_kernel" \
  crates/infer-cuda-kernels/src/ffi/kv.rs \
  crates/infer-cuda-kernels/csrc/kv/kv_cache_to_paged.cu
```

Expected: one FFI declaration plus one CUDA kernel/wrapper pair.

```bash
rg -n "Coordinator|HostPinnedPool|LocalCudaTransport|hit_count|tier_location|soft_pin_until|fingerprint" \
  infer/src/{kv_tier.rs,prefix_cache.rs,types.rs} infer/src/kv_tier
```

Expected:
- `Coordinator`, `HostPinnedPool`, and `LocalCudaTransport` are exported.
- `prefix_cache::Node` carries the tier-aware metadata fields.

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
- [ ] No new CUDA-only linker/runtime failure is introduced by the page-size
      lift or the M3a scaffolding.
- [ ] Golden outputs remain unchanged.

---

## 4 · BF16 page-size gate

Run a focused throughput comparison on the same host / same model / same
flags.

If the host does **not** already have a pre-M0.3 baseline, first record one
from the last known page-size-1 BF16 commit and save it as `--label page1`.

Then run the new build:

```bash
python3 scripts/bench_throughput_sweep.py \
  --label page16 \
  --model-path models/Qwen3-4B
```

Acceptance:
- [ ] The sweep completes without CUDA faults.
- [ ] BF16 decode/prefill throughput does not show an obvious regression
      versus the same-host `page1` baseline.
- [ ] No short-context crash or metadata-shape error appears in logs.

If the sweep exposes a regression, record both snapshots before debugging:
- `docs/experience/wins/YYYY-MM-DD-bench-page1-<host>.md`
- `docs/experience/wins/YYYY-MM-DD-bench-page16-<host>.md`

---

## 5 · M3a structural smoke

This batch is intentionally structural. On a CUDA host, acceptance means the
transport/pool scaffolding links and the pure-Rust tests still pass.

```bash
cargo test -p infer --release kv_tier
cargo test -p infer --release prefix_cache
```

Acceptance:
- [ ] All `kv_tier` tests pass on the CUDA host.
- [ ] `prefix_cache` tests still pass with the new node metadata fields.
- [ ] The build carries `crossbeam-channel` and the new `kv_tier` modules
      without linker/runtime fallout.

Optional manual smoke if you want an explicit API sanity check on the host:

```bash
rg -n "LocalCudaTransport requires at least one transfer op|GPU required: LocalCudaTransport poll is a structural stub" \
  infer/src/kv_tier/transport/local_cuda.rs
```

Expected: the current host-side implementation is still a structural stub;
real `cudaMemcpyAsync` behavior remains M3b work.

---

## 6 · Sign-off checklist

M0.3 + M3a are accepted when all of the below are true:

- [ ] Static sanity checks passed.
- [ ] Full build/test gate passed on CUDA.
- [ ] BF16 page-size throughput sweep completed and was compared against the
      same-host `page1` baseline.
- [ ] `kv_tier` / `prefix_cache` structural smoke passed on CUDA.
- [ ] A win note was written under `docs/experience/wins/` with:
      environment, commands, raw outputs or linked artifacts, and the
      `page1` vs `page16` comparison.

After sign-off, update:
- `docs/projects/tiered-kv-cache.md` — M0.3/M3a status from
  "local impl done; remote validation pending" to "accepted".
- `docs/plans/tiered-kv-cache-tasks.md` — mark the remote checklist done.
- `docs/index.md` — add the new win note if it is not already listed.
