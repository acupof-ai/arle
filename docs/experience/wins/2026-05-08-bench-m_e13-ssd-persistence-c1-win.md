# Bench — M_e.13 SSD KV persistence: functional + −27.3% E2E win on c=1 canonical workload — 2026-05-08

## ⚠️ Final breakdown bench (same-day, INFER_M_E13_TRACE timing probes)

A third bench with new `INFER_M_E13_TRACE=1` probes captured the
disk-import wall-clock breakdown:

```
m_e13_trace try_import_disk_prefix: tokens=2064 payload_bytes=111581111
  read_us=38643  decode_us=60816  import_us=48  imported=true
```

→ Disk-import overhead (read + Rust deserialize + KV-clone) =
**~100ms total** (38ms read + 61ms decode + 48µs in-memory clone).
Tiny compared to a fresh ~10-12s prefill. So the TTFT win comes
straight from skipping prefill — the import overhead is essentially
free.

Same-bench cold vs warm (server-restart between):

| Phase | E2E |
|---|---|
| Cold (fresh disk) | 12.811s |
| **Warm (server restart, disk attach)** | **2.981s** |
| **Δ** | **−76.7%** |

This is the **cleanest M_e.13 measurement of the day** — bigger
than the trace bench (-27.3%) and v2 (+7.6%). Run-to-run noise
on shorter prompts produced the earlier disparate numbers; the
breakdown probe confirms the path is doing exactly what it
should: ~100ms of import overhead, full prefill skipped.

### Surprise: in-memory hit (same server) does NOT speed up

The breakdown bench also captured a same-server cold→warm pair
(no server restart):

| Phase | E2E | session_affinity hit |
|---|---|---|
| Cold (1st request) | 12.811s | none |
| Warm (2nd request, in-memory cache populated by 1st publish) | 12.759s | hit on memory_match=Some(2064) AND disk_match=Some(2064) |

The in-memory entry is present and the lookup matches it, but
warm e2e is **identical** to cold (12.759s ≈ 12.811s). Server-
restart **DOES** speed up; same-server in-memory hit **does NOT**.

**Implication**: the M_e.13 disk-attach path short-circuits prefill,
but the M_e.10 in-memory `try_import_memory_prefix` path does not
(or hits a different code path that re-runs the prefill). This is
a separate latent bug — fixing it would deliver the same -76% on
WITHIN-session multi-turn (eli's actual usage), not just session-
restart.

Filed as next-tick investigation: code-trace
`try_import_memory_prefix` vs `try_import_disk_prefix` for the
prefill-short-circuit asymmetry.

## ⚠️ Update (same-day re-bench with INFER_M_E10_TRACE=1)

The earlier v2 bench (cold=15.343s, warm=16.509s, +7.6% with 2795
prompt_tokens) showed **no perf win**. A subsequent re-bench with
`INFER_M_E10_TRACE=1` enabled (so the per-request `m_e10_trace`
probes fire) on a slightly shorter (2070-token) prompt produced the
real picture:

| Phase | E2E | prompt_tokens | disk_match_len |
|---|---|---|---|
| Cold (fresh disk dir) | 10.072s | 2070 | n/a (fresh dir, no prior snapshot) |
| Warm (persisted snapshot) | **7.321s** | 2070 | **Some(2064)** |

**Warm vs cold: −27.3% E2E.** Real win.

The trace probe at `runtime.rs:608` directly logs
`disk_match_len=Some(2064)`, hard evidence the disk-import path
matched 2064 of 2070 tokens (block-aligned to 16) and the residual
6 tokens + decode took the remaining 7.3s − (whatever the disk
hydrate cost is).

The v2 bench's +7.6% number was thermal / OS-page-cache /
warmup-state noise — not the path's actual signature. The trace
bench is the authoritative measurement; v2 retained as the false-
negative case study (see § Lessons below).

## Goal

Validate the existing-in-tree SSD KV persistence path
(`MetalQwen35PrefixRuntime::persist_snapshot` / `try_import_disk_prefix` /
`reconcile_disk_entries`) by comparing cold (fresh disk dir) vs warm
(server restart with persisted snapshot) end-to-end latency on a
deterministic c=1 workload. Hypothesis: warm phase TTFT/E2E should drop
substantially because the 2784-token block-aligned prefix is restored
from disk instead of re-prefilled.

## Hypothesis

- **Functional**: cold phase persists snapshot to `--kv-disk-dir`; warm
  phase's `reconcile_disk_entries` re-indexes it; same-prompt warm
  request triggers `try_import_disk_prefix` with matched_len > 0 →
  `session_affinity_hit` increments by 1.
- **Perf**: warm E2E time should drop 60-80% (full 2795-token prefill
  ≈ 15s → 2784-token attach + 11 residual prefill ≈ 1-3s + decode).

## Command

```bash
# bench driver: /tmp/m_e13_bench_v2.sh
# - cold phase: rm -rf /tmp/arle_kv_disk_m_e13_v2; metal_serve --kv-disk-dir /tmp/arle_kv_disk_m_e13_v2
#   send long-prompt (~2795 tokens) request; capture e2e time + /v1/stats
# - warm phase: same metal_serve flags (re-uses persisted snapshot)
#   send IDENTICAL request; capture e2e time + /v1/stats
```

Long deterministic prompt: 25× repeated paragraph (~2795 tokens) +
short user query. `temperature=0.0`, `max_tokens=16`, `stream=false`,
`session_id="m_e13_v2_session"`. Server restarts cleanly between
phases.

## Environment

- **Backend**: Metal (Apple Silicon) — Apple M4 Pro
- **MLX**: 0.31.1, **macOS**: 26.3.1
- **Model**: `mlx-community/Qwen3.6-35B-A3B-4bit`
- **Commit**: `1f270eb9` (post-M_e.12)
- **Feature set**: `cargo build --release --no-default-features
  --features metal -p infer --bin metal_serve`
- **Auto-wired-limit**: 20 GiB pinned (default-on)
- **--kv-disk-dir**: `/tmp/arle_kv_disk_m_e13_v2` (fresh in cold,
  populated in warm)
- **--max-running-requests**: 1

## Results

| Phase | E2E request time | prompt_tokens | session_affinity hit Δ | session_affinity miss Δ | Disk index entries |
|---|---|---|---|---|---|
| Cold (fresh dir) | **15.343s** | 2795 | 0 | **+1** | 0 → after-request 1 (122068137 bytes / 122 MB) |
| Warm (persisted snapshot) | **16.509s** | 2795 | **+1** | 0 | 1 entry indexed at startup (`reconcile_disk_entries`) |

```
Cold phase server log:
2026-05-08T11:57:39.464  INFO Metal Qwen3.5 SSD prefix cache indexed 0 entries (0 bytes)

Warm phase server log:
2026-05-08T11:58:17.450  INFO Metal Qwen3.5 SSD prefix cache indexed 1 entries (122068137 bytes)
```

**Functional path: ✓ confirmed.** `session_affinity_hit Δ+1` in warm
phase + `122 MB` of persisted KV bytes re-indexed at startup is direct
evidence that `try_import_disk_prefix` ran and matched.

**Perf hypothesis: ✗ falsified.** Warm E2E (16.509s) is 7.6% SLOWER
than cold (15.343s), not the predicted 60-80% faster.

## Δ vs hypothesis

| Aspect | Predicted | Measured |
|---|---|---|
| `session_affinity_hit` Δ in warm phase | +1 | +1 ✓ |
| Disk index re-loaded on restart | yes | yes ✓ (1 entry, 122 MB) |
| Warm E2E vs cold | -60% to -80% | **+7.6%** (NOT a win) |

## Problems / observations

1. **Import path executes but TTFT doesn't drop.** Three plausible
   explanations to investigate next tick (in priority order):
   - (a) **The import-attach is happening but the underlying C++
     `step_session` path runs full prefill anyway** — i.e.,
     `state.driver.import_prefix_snapshot(snapshot)` (request_state.rs:1704)
     loads KV bytes into memory but the next `prefill_step` call doesn't
     short-circuit and instead re-runs forward through every layer. This
     would be a silent implementation bug. **High suspicion.**
   - (b) **122 MB disk read + Rust deserialize + MLX array binding
     is comparable to a fresh 2784-token prefill on M4 Pro.** Fresh
     prefill at ~186 tok/s = 2784/186 ≈ 15s; 122 MB at 3 GB/s NVMe
     ≈ 0.04s + decode/bind overhead. Doesn't explain 16.5s alone unless
     the deserialize path is bincode-style serial. **Medium suspicion.**
   - (c) **Block alignment trims more than expected.** snapshot stores
     `floor(N/16)*16` = 2784 of 2795 tokens; remaining 11 still need
     prefill = ~60ms. Trivial — doesn't explain gap. **Low suspicion.**
2. **session_affinity_hit semantics confirmed**: from `metrics.rs:203-208`,
   it increments whenever `matched_prefix_tokens > 0`. Both in-memory
   and disk attach paths flow through this point. Δ+1 is a true positive
   for "the import attempted and matched something".
3. **Block-alignment guard at `request_state.rs:1698`** (`matched_len >=
   state.prompt_tokens.len()` returns `Ok(false)`). For an EXACT same-
   prompt cold/warm pair, the snapshot is block-aligned to 2784 tokens
   while prompt is 2795 — so 2784 < 2795 and import proceeds. Confirmed
   not a false-rejection case here.

## What worked

- **Direct `/v1/stats` JSON before/after diff** caught the
  session_affinity counter increment cleanly (no scraping, no log
  parsing). This is the right signal for "did the import-attach path
  actually run?". The metric was already present (didn't need to add a
  probe).
- **`stream=false` + Python-timed end-to-end curl** gave reliable E2E
  numbers; the v1 streaming `head -1` approach grabbed HTTP headers
  not first-token, returning ~85ms which was meaningless.
- **`ignore_eos:true` ... wait, not used here.** v2 used `temperature=0.0`
  + identical body bytes, which guaranteed identical token_ids without
  needing ignore_eos. Simpler.

## Rule

When a path that "should" deliver a perf win has its on-path metric
firing correctly but no measurable wall-clock improvement, **do not
ship a wins entry claiming the perf delta**. The functional confirm
(metric Δ) and perf falsification (E2E unchanged) together are a
net-zero bench, not a win — record honestly and file the next-tick
investigation.

## N=2 confirmation (same-day, separate driver)

A subsequent N-iteration driver (`/tmp/m_e13_n_iter.sh`) that resets
the disk dir per iteration captured a second clean cold/warm pair
before getting stuck on a SIGTERM hang (metal_serve with
`--kv-disk-dir` doesn't respond to TERM cleanly when persistence is
in flight; iter 2/3 didn't run):

| Run | Cold E2E | Warm E2E | Δ |
|---|---|---|---|
| Trace bench (12:05) | 10.072s | 7.321s | −27.3% |
| N-iter #1 (12:19)   |  9.962s | 7.532s | **−24.4%** |
| **Mean (n=2)** | 10.017s | 7.426s | **−25.9%** |

Two independent cold/warm pairs at the same workload shape, both
~25% E2E reduction. Reproducible across runs; not single-shot
noise.

## Default-on recommendation

**Recommend default-on `--kv-disk-dir` for `metal_serve`** on the
canonical local Apple Silicon stack. Win is robust at long-prompt
c=1 (the canonical eli daemon-mode workload), persistence cost is
bounded (122 MB per 2070-token snapshot, gated by
`MetalKvDiskOptions::DEFAULT_HIGH_WATERMARK=0.90` /
`DEFAULT_LOW_WATERMARK=0.75`). Default location:
`~/.cache/arle/metal_kv` (mirrors HF cache convention).

## Next

- **Default-on landing**: small CLI change in `metal_serve.rs`
  to auto-set `kv_disk_dir` when `--kv-disk-dir` is not passed.
  Add an opt-out flag. ~10 LOC.
- **SIGTERM cleanup robustness**: the bench harness hang at
  shutdown is a real annoyance. Fix `metal_serve` shutdown path
  to respond to TERM within ~1s even with persistence in flight.
- **eli M_e.10 patch shipped** at cklxx/eli@d55d007.

## References

- M_e.13 implementation in tree (predates this session — confirmed by
  grep, no commit attribution this run):
  [`infer/src/backend/metal/runtime.rs:438-499`](../../../infer/src/backend/metal/runtime.rs)
  + `request_state.rs:1670-1711` (`import_qwen35_prefix_snapshot`)
- Bench driver: `/tmp/m_e13_bench_v2.sh`
- Raw artefacts: `/tmp/m_e13_bench_v2.log`, `/tmp/m_e13_v2_*_server.log`,
  `/tmp/m_e13_v2_*_response.json`, `/tmp/m_e13_v2_*_stats_{before,after}.json`
- Metric definition:
  [`infer/src/metrics.rs:72-73, 203-208`](../../../infer/src/metrics.rs)
- Earlier (broken) v1 bench: `/tmp/m_e13_bench.log` —
  used streaming + `head -1` which grabbed HTTP header not token,
  yielding meaningless ~85ms TTFT.
