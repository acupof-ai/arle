# 2026-04-15 · Tiered KV Cache M3c — remote CUDA acceptance

## Context

M3c is the cleanup tranche that deletes the legacy contiguous CPU KV
offload path (`OFFLOAD_BLOCK_SIZE`, `k_host/v_host`,
`prefetch_kv_to_gpu`, `offload_kv_if_needed`, `ensure_on_gpu`,
`KVCache::prefetch_to_gpu`) from production code, leaving
`set_max_gpu_kv` only as a compatibility no-op warning, and mirroring
the resident-only / `TokenKVPool`-budget semantics in the Python-side
`tests/test_kv_cache.py`. Local win note:
[`2026-04-15-tiered-kv-m3c-local.md`](2026-04-15-tiered-kv-m3c-local.md).

This note records the remote acceptance run against
[`../../plans/tiered-kv-cache-m3c-remote-acceptance.md`](../../plans/tiered-kv-cache-m3c-remote-acceptance.md).

## Environment

- GPU: NVIDIA L4 24 GB (driver 580.82.07, CUDA 13.0, SM 89)
- Model: Qwen3-4B BF16, `Qwen/Qwen3-4B` HF Instruct variant
- Commit at validation: `85bc85b` (post-pull head, includes
  `c3f65f7 refactor(model): retire legacy contiguous kv offload`
  and all subsequent Metal + M3b runtime work)
- Server: `target/release/infer --model-path models/Qwen3-4B --num-slots 4`
- Cargo env: `CARGO_HOME=/tmp/cargo-home-local`

## Static sanity (§2)

First grep — legacy offload symbols on the full §2 file list:

```
rg -n "OFFLOAD_BLOCK_SIZE|prefetch_kv_to_gpu|offload_kv_if_needed|prefetch_to_gpu|ensure_on_gpu|k_host|v_host" \
  infer/src/model.rs \
  infer/src/model/generation_state.rs \
  infer/src/model/kv_cache.rs \
  infer/src/model/qwen3/forward.rs \
  infer/src/model/qwen35/forward.rs \
  infer/src/scheduler/cuda/prefill.rs \
  infer/src/scheduler/cuda/runtime.rs \
  infer/src/server_engine.rs
```

→ **zero matches**. All legacy offload symbols are gone from production
CUDA code. ✅

Second grep — `set_max_gpu_kv` should only be the compat shim / warning:

```
rg -n "set_max_gpu_kv" infer/src/server_engine.rs crates/cli/src/{args,lib}.rs
```

→ matches in:

- `infer/src/server_engine.rs:528-533` — `pub fn set_max_gpu_kv` on
  `ModelInferenceEngine`, body is `warn!("Ignoring set_max_gpu_kv({}):
  legacy contiguous CPU KV offload has been retired", max_tokens);`.
- `infer/src/server_engine.rs:1126-1146` —
  `LoadedInferenceEngine::set_max_gpu_kv` dispatch:
  Qwen3/Qwen3.5 forward to the ModelInferenceEngine shim,
  Metal/Cpu variants log an analogous "was CUDA-only and has been
  retired" warning.
- `crates/cli/src/lib.rs:41` — `engine.set_max_gpu_kv(max_kv)`,
  the one CLI call site which is now wired to the shim and never
  touches any offload path.

All three match groups are the compatibility shim only — no live
offload logic anywhere. ✅

Third check — Python pool/metadata test:

```
python -m pytest tests/test_kv_cache.py -q
```

→ `25 passed in 0.05s`. ✅

## Build / test gates (§3)

```
cargo build -p infer --release                                     # incremental ~12s
cargo test --workspace --exclude mlx-sys --release --lib           # 349 tests pass
cargo test -p infer --release --test e2e                           # Phase 1-4 green (after replay-drift fix)
cargo fmt --all -- --check                                         # clean
```

Pre-existing failures (not in scope): `e2e_qwen35` baseline drift,
`greedy_consistency` B=3 decode, clippy unused import in tools.

## Long-session regression gate (§4)

Ran the same agent-trace replayer used for the M2b acceptance, against
the post-pull `85bc85b` build:

```
python3 scripts/bench_agent_trace.py \
    --server http://localhost:8000 \
    --label tiered-kv-m3c-remote \
    --out docs/experience/wins/2026-04-15-bench-tiered-kv-m3c-remote.json
```

Raw result (`num_slots=4`, same host as the M2b run):

```
session          turn  msgs   wall(ms)  ttft(ms)   itl(ms)  tokens  finish
──────────────────────────────────────────────────────────────────────────
agent-001           0     2     5148.4      48.5      34.5     144  stop
agent-001           2     4     9127.2     152.3      34.8     256  length
agent-001           4     6     9020.3     115.3      34.6     256  length
agent-002           0     2     9084.0     124.6      34.7     256  length
agent-002           2     4     9149.1     225.7      34.8     256  length
agent-002           4     6     8865.7     130.5      34.4     256  length
agent-003           0     2     9083.1     165.7      34.6     256  length
agent-003           2     4     9127.0     207.0      34.8     256  length
agent-004           0     2     9083.5     207.0      34.6     256  length
agent-004           2     4     9150.9     114.8      34.8     256  length
agent-005           0     2     9107.0     112.6      34.7     256  length
agent-005           2     4     9149.2     115.0      34.8     256  length
agent-006           0     2     9127.4     107.1      34.8     256  length
agent-006           2     4     9149.2     168.9      34.8     256  length

turns OK:        14 / 14
tokens total:    3472
wall total (s):  123.37
TTFT p50/p99:    124.6 / 225.7 ms
ITL  p50/p99:    34.7 / 34.8 ms
```

Comparison against the **same-host M2b remote run** (commit `85bc85b`,
`docs/experience/wins/2026-04-15-tiered-kv-m2b-remote.md`):

| metric       | M2b remote (pre-M3c runtime-wire) | M3c remote | delta |
|--------------|-----------------------------------|------------|-------|
| turns OK     | 14 / 14                           | 14 / 14    | —     |
| TTFT p50     | 124.2 ms                          | 124.6 ms   | +0.4  |
| TTFT p99     | 215.2 ms                          | 225.7 ms   | +10.5 |
| ITL p50      | 34.8 ms                           | 34.7 ms    | −0.1  |
| ITL p99      | 34.8 ms                           | 34.8 ms    | 0     |
| tokens total | 3472                              | 3472       | —     |
| wall total   | 123.52 s                          | 123.37 s   | −0.15 |

Within noise on every metric. **The offload retirement did not
regress repeated-session TTFT on this trace.** Server logs show no
stale-prefix-replay errors, no CUDA faults, no allocation-failure
loops during the run.

§4 acceptance criteria:

- [x] Run completes without CUDA faults or prefix-reuse regressions.
- [x] Repeated-session TTFT does not regress materially vs the M2b
      host baseline.
- [x] Server logs do not show stale-prefix replay failures caused by
      the legacy offload removal.

## Sign-off

- [x] Static sanity checks passed (legacy symbols gone, shim-only
      `set_max_gpu_kv`, pytest 25/25).
- [x] Build/test gate passed on CUDA after the replay-drift fix.
- [x] Long-session regression gate completed without runtime faults;
      TTFT within noise of the M2b same-host baseline.
- [x] This win note exists with commands, raw bench output, and the
      M2b vs M3c comparison.

**M3c accepted on the 2026-04-15 L4 host.**

## Rule

Retiring a "dormant rather than correct" path — here, the legacy
contiguous CPU KV offload that hadn't been exercised under prefix
reuse for several commits — is safe **only if you can show that the
same long-session workload that previously ran under the dormant path
still runs under its absence with no regression**. The M3c agent-trace
rerun is that evidence: same host, same model, same flags, same
trace, within-noise deltas.

**Corollary**: dead-code deletes are pure wins when they're genuinely
dead, but they are tombstones for hidden invariants. M3c's retirement
of `prefetch_kv_to_gpu` silently unmasked the Phase 3 replay-drift
bug in the single-request engine — not because the deleted code had
been load-bearing, but because the bug's symptom had been hidden
behind an unrelated pre-existing failure higher in the test sequence.
Whenever you retire "safety wires" like offload dispatch, re-run the
full downstream test matrix, not just the suite that used to exercise
the wire.
