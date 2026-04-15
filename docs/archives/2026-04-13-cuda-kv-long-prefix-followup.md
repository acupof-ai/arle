> **Archived 2026-04-15** — the bug captured in this doc lives in the
> **legacy contiguous CPU offload** path (`KVCache::offload_if_needed`
> at `infer/src/model/kv_cache.rs:517-590`, `OFFLOAD_BLOCK_SIZE = 64`).
> That entire code path is now slated for full deletion in
> [`../projects/tiered-kv-cache.md`](../projects/tiered-kv-cache.md)
> §6 M3c (see also §3 fact 5 and §8 pitfall 13). Once M3c lands, the
> bug is "fixed by deletion": the new paged-pool tiering does not
> share the failure mode, and the legacy offload that triggered the
> panic no longer exists.
>
> One M2b note: the 2026-04-15 M2b batch added
> `GenerationState::prefetch_kv_to_gpu()` which currently *uses* the
> legacy offload path as a correctness bridge before prefix reuse
> reads contiguous KV. M3c will need to delete that hook too — see
> the tiered-kv project doc §6 M3c sub-PR scope.
>
> Preserved for the root-cause analysis (`max_seq_len` accounting
> mismatch), which informs M3c's invariant choice.

---

# 2026-04-13 CUDA KV Long-Prefix Follow-up

## What Landed Today

- Pushed `3af127d` `fix(cuda): harden prefix cache and paged decode state`
- Pushed `c28d5ad` `fix(models): stabilize batched decode sampling fallback`
- Revalidated CUDA lib tests after those fixes:
  - `cargo test -p infer --lib -- --test-threads 1`
  - Result: `246 passed; 0 failed; 11 ignored`
- Revalidated KV prefix correctness:
  - `python3 scripts/verify_kv_cache.py http://127.0.0.1:8000`
  - Result: cold vs warm all `PASS`

## New Issue Found During Long-Prefix Perf Validation

While running a long-prefix KV latency benchmark, the server hit a real bug in the KV offload path instead of producing a usable measurement.

Observed failure in server log:

```text
thread '<unnamed>' (...) panicked at infer/src/backend/cuda/tensor.rs:250:9:
copy_region_to_host: offset 3080192 + len 4148224 exceeds buffer len 4194304
```

Context from the same run:

- Server was started with the default auto-sized `max_seq_len=4096`
- Benchmark request reached `prompt_tokens=7059`
- Cleanup then tried to offload `3008` tokens after the request completed
- The panic happened inside `KVCache::offload_if_needed()`

## Repro

1. Start the normal server:

```bash
RUST_LOG=info CUDA_HOME=/usr/local/cuda ./target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8000
```

2. Send a very long shared-prefix chat request. The exact prompt used today was roughly `44k` chars / `7059` prompt tokens.

3. Watch the server log during cleanup. The background scheduler thread panics in `copy_region_to_host()`.

## Current Root-Cause Read

This is not a fake benchmark failure. The current KV/offload accounting is inconsistent for prompts that exceed the configured total sequence length.

The likely chain is:

1. `KVCache.max_seq_len` is the total contiguous capacity allocated on GPU per layer.
2. `KVCache.seq_len` is allowed to grow past that capacity during long-prefill / decode.
3. `KVCache::offload_if_needed()` assumes the GPU buffer currently contains `0..gpu_tokens` contiguously.
4. Once `seq_len > max_seq_len`, the shift/copy math reads past the end of the contiguous buffer and the assert in `DeviceVec::copy_region_to_host()` trips.

The key mismatch to resolve tomorrow:

- `max_seq_len` is being treated as a hard allocation boundary by the storage layer.
- The runtime path still behaves as if offload can rescue sequences that already exceeded that boundary.

Those two things cannot both be true.

## Why This Matters

- The shorter KV correctness path is fixed and validated.
- The long-prefix perf path is not trustworthy until this boundary condition is made self-consistent.
- A “just reject it” patch would hide the accounting bug, not explain it.

## Tomorrow's Fix Plan

1. Audit every place that advances `KVCache.seq_len` and every place that assumes `seq_len <= max_seq_len`.
2. Decide the intended invariant explicitly:
   - Either `max_seq_len` is a hard total-context limit.
   - Or long-context/offload must support `seq_len > max_seq_len`, which requires more than the current contiguous-buffer scheme.
3. Patch `KVCache` so the chosen invariant is enforced without panic.
4. Add regression coverage for:
   - long prefill near the configured limit
   - overflow past the configured limit
   - offload + cleanup after a long request
5. Re-run long-prefix perf with a server configuration that actually matches the intended benchmark target, likely with a larger explicit `--max-seq-len` and fewer slots.

## Bench Commands To Re-Run After The Fix

KV correctness:

```bash
python3 scripts/verify_kv_cache.py http://127.0.0.1:8000
```

Service throughput:

```bash
python3 scripts/bench_throughput_sweep.py --url http://127.0.0.1:8000 --label cuda-l4-opt --quick
```

Long-prefix KV latency:

- Re-run the custom long-prefix benchmark only after the `max_seq_len` / offload invariant is fixed and the server is started with a matching long-context configuration.
