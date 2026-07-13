# Durable CUDA KV recall survives process restart — H20 pod, 2026-07-13

## SLO-shape probed? N

The canonical 4096-input/256-output GuideLLM sweep is a regression guard, not an
8K SLO probe. This change is a correctness repair with no default flip.

## Roofline check

Not applicable. The changed path runs only while attaching the disk tier and
rebuilding its host index; it adds no decode or prefill arithmetic.

## Goal

Prove that issue #136's durable CUDA KV store is discoverable and byte-readable
from a different process, and that dense Qwen3 attaches the durable lane.

## Hypothesis

A stable namespace keyed by weights epoch, KV format, TP world/rank, and page
bytes lets process B restore process A's manifest while rejecting incompatible
geometry and concurrent owners.

## Command

```bash
scripts/pod.sh sync crates/kv-native-sys/src/lib.rs \
  crates/kv-native-sys/src/kv_tier.rs \
  crates/infer-cuda/src/executor/qwen.rs \
  crates/infer-cuda/src/executor/qwen35.rs
scripts/pod.sh build issue136-kv -p kv-native-sys --tests

# Inside the same ordinary sglang-test pod, with ROOT=/host/arle-issue136-proof:
ARLE_KV_TIER_CROSS_PROCESS_MODE=write \
ARLE_KV_TIER_CROSS_PROCESS_ROOT="$ROOT" "$TEST" --exact \
  kv_tier::tests::durable_manifest_round_trips_disk_index_across_processes
ARLE_KV_TIER_CROSS_PROCESS_MODE=read \
ARLE_KV_TIER_CROSS_PROCESS_ROOT="$ROOT" "$TEST" --exact \
  kv_tier::tests::durable_manifest_round_trips_disk_index_across_processes

POD_TREE=/host/arle-build-issue136 scripts/pod.sh run issue136-serve1 0 -- \
  serve --backend cuda --model-path /host/Qwen3-4B --port 18036 \
  --kv-recall --kv-cache-dtype bf16 \
  --kv-disk /host/arle-issue136-serve --kv-disk-limit 1GiB --kv-dram 0 \
  --max-running-requests 1 --max-total-tokens 8192

GUIDELLM_OUTPUTS="json csv" scripts/bench_guidellm.sh issue136-kv-durable \
  --target http://localhost:18036 \
  --model /host/Qwen3-4B --processor /host/Qwen3-4B
```

## Environment

- Backend: CUDA, BF16 KV, recall enabled, DRAM tier disabled.
- Hardware: NVIDIA H20, 97,871 MiB; GPU 0 isolated for this run.
- Model: `/host/Qwen3-4B`, one rank.
- Commit: `751d94590` (built from clean tree at `eb256db02`).
- Feature set: `cargo build --release --features cuda --bin arle`.
- Disk: `/host`, persistent node filesystem.

## Results

### Cross-process disk reuse

| Gate | Result |
|---|---|
| Writer PID | `4091029` |
| Reader PID | `4091031` |
| Prior pages read by reader | 3/3 |
| New page inserted/read after reload | key 4, byte-exact |
| Concurrent owner | rejected by nonblocking exclusive lock |
| Durable namespace | `arle-kv-recall-epoch-A-format-1-world-4-rank-3-page-8` |
| Persistent files before cleanup | `kv.mmap` 32 B; `manifest.kvm` 64 B; `owner.lock` 0 B |

### Dense Qwen3 attach

The first server boot logged:

```text
KV recall restored 0 pages from /host/arle-issue136-serve/arle-kv-recall-st-426b3aeaf79040f9-format-1-world-1-rank-0-page-2359296
KV tiers: dtype=bf16 | ... | L3 root=/host/arle-issue136-serve cap 1073741824B/rank | features: prefix,recall
```

During GuideLLM, `/v1/stats` reported `disk_pages=244`; before shutdown the
backing mmap was 1,073,479,680 bytes and the manifest was 9,884 bytes. The
second server process logged:

```text
KV recall restored 246 pages from /host/arle-issue136-serve/arle-kv-recall-st-426b3aeaf79040f9-format-1-world-1-rank-0-page-2359296
```

A post-restart `/v1/completions` request completed normally (5 prompt tokens,
8 generated tokens). This proves the dense executor attached the prior mmap and
manifest without preventing subsequent service.

### GuideLLM

The canonical sweep did not complete. Its first 4096-input/256-output request
reached 203 generated tokens, then the coordinator reported a lockstep stall for
30 seconds and `/v1/stats` timed out. No throughput delta is claimed. The run
still exercised the changed tier: `disk_pages` rose from 0 to 244 before the
restart restored 246 manifest records.

## Problems

- The shared pod build tree was locked by another container. The documented
  per-agent `POD_TREE` path was used instead; its clean Git history contains
  `751d94590`, and `strings target/release/arle` contains the restore log symbol.
- Durable records use `(slot, logical_page)` keys. This repair proves durable
  byte/index reuse across restart; it does not persist scheduler session or page
  tables, so it does not claim that generation resumes an old session.
- The canonical GuideLLM request stalled after 203 output tokens. It changed no
  benchmark variables beyond enabling the affected feature, but no recall-off
  control was run in this issue closure, so the cause is explicitly unassigned.

## Learnings

- A restart-safe directory is insufficient without geometry isolation, mmap free
  slot reconstruction, stale-manifest rejection, and a single-writer lock.
- Cross-process PID evidence plus a byte-exact read is the minimum durable-store
  gate; directory existence alone proves nothing.

## Delta vs baseline

No delta: the canonical run stalled and no prior entry uses this model, flags,
and disk geometry. The durable-restart correctness verdict is independent of
throughput.

## Artefacts

- Pod build log: `/root/build-issue136-arle-clean.log`.
- Server log: `/root/run-issue136-serve1.log`.
- Raw GuideLLM output (incomplete):
  `bench-output/2026-07-13-issue136-kv-durable/`.
