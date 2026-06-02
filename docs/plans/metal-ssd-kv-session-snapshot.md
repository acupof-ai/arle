# Metal SSD KV Session Snapshot Plan

## Goal

Use SSD only as a bounded prefix/session snapshot tier for Metal Qwen3.6.
Active decode KV must stay resident in unified memory. SSD reads happen once
per request on a prefix hit, then the snapshot is promoted to the in-memory LRU.
SSD writes happen after useful work has already been paid for, through a
background writer.

This plan is based on the local SSD measurement in
`docs/experience/wins/2026-06-02-metal-ssd-kv-throughput-budget.md`:

- New-token KV append needs only about 1.3-1.6 MiB/s at the measured decode
  rate.
- A 12k pure-KV prefix snapshot is about 240 MiB and reads in about 58 ms on
  the conservative 4.05 GiB/s path.
- Reading the full active KV history from SSD on every decode step is not
  viable: 4k already needs 6.30 GiB/s; 12k needs 15.74 GiB/s before import and
  Metal overhead.
- Small-block random reads are too slow for per-layer paging: 64 KiB random
  read is about 0.121 ms median / 0.195 ms p95.

## Strategy

The runtime should have three residency states:

```text
T0 resident snapshot  ->  MLX KV+GDR arrays held in memory LRU
T2 disk snapshot      ->  full prefix/session snapshot on SSD
active request state  ->  imported snapshot + suffix prefill + decode, all in memory
```

Request flow:

```text
tokenize
  -> longest in-memory snapshot hit
  -> else longest SSD snapshot hit
  -> import snapshot once
  -> prefill only the unmatched suffix
  -> decode entirely in memory
  -> publish completed session snapshot asynchronously
```

Do not build:

- Per-token SSD KV reads.
- Per-layer SSD page-in.
- Direct attention over SSD-backed KV.
- Synchronous per-token `fsync`.

## Current Code Map

### Already Good

- `infer/src/backend/metal/runtime.rs:290` defines the Metal prefix block size
  and SSD persist gate defaults.
- `infer/src/backend/metal/runtime.rs:295` documents the current gate:
  persist only when paid prefill cost is larger than predicted readback cost
  with safety margin.
- `infer/src/backend/metal/runtime.rs:387` owns `MetalDiskPrefixIndex`, a
  shared LRU index with byte accounting and high/low watermarks.
- `infer/src/backend/metal/runtime.rs:659` runs a dedicated SSD writer thread.
  Writes are already off the serving thread after payload encoding.
- `infer/src/backend/metal/runtime.rs:901` wires `DiskStore`, model
  fingerprint, disk max bytes, watermarks, and `fsync_each_block`.
- `infer/src/backend/metal/runtime.rs:1009` checks memory and disk snapshots on
  request admission.
- `infer/src/backend/metal/runtime.rs:1398` reads a disk snapshot, decodes it,
  imports it, then promotes it into the in-memory snapshot LRU.
- `infer/src/backend/metal/request_state.rs:316` defines
  `Qwen35PrefixSnapshot` as token ids plus KV and GDR arrays.
- `infer/src/backend/metal/request_state.rs:348` encodes snapshots with
  metadata and body checksums.
- `infer/src/backend/metal/request_state.rs:456` decodes snapshots from disk
  and validates the model fingerprint.
- `infer/src/backend/metal/request_state.rs:4818` imports KV+GDR arrays into
  the live Qwen3.5/Qwen3.6 C++ step driver.

### Main Gap

The current publish path snapshots only the prefilled prompt cursor:

- `infer/src/backend/metal/runtime.rs:1125` calls
  `publish_prompt_prefix()`.
- `infer/src/backend/metal/runtime.rs:1140` calls
  `export_qwen35_live_prefix_snapshot()`.
- `infer/src/backend/metal/request_state.rs:1743` sets
  `live_len = state.prompt_cursor`.

That means a multi-turn agent session can reuse the user prompt, but it does
not persist `prompt + assistant generated tokens` as the next-turn prefix.
For chat/agent workloads, the useful next request usually extends the complete
previous transcript, not just the previous prompt.

### Secondary Gaps

- `ActiveMetalRequest` keeps `pending_token_ids` only for streaming deltas
  (`infer/src/backend/metal/runtime.rs:112`); it does not retain a completed
  generated-token vector for snapshot keys.
- `ResumableRequestState` tracks `generated_tokens` count and `last_token`
  (`infer/src/backend/metal/request_state.rs:123`), but not generated token
  ids.
- `record_sampled_token()` commits the token to runtime state
  (`infer/src/backend/metal/request_state.rs:256`) but does not expose an
  append history.
- Disk eviction is pure LRU. It does not consider saved prefill time per byte.
- Disk read/write/import timing is currently available through env-gated logs,
  not stable metrics.

## Implementation Plan

### Phase 0: Lock The Contract

Add comments/tests that make the core rule explicit:

- SSD snapshots may be imported only before suffix prefill starts.
- After import, decode must not issue SSD reads.
- A disk hit must promote to memory when import succeeds.
- Snapshot keys are token-id prefixes, not session ids. `session_id` may bias
  eviction later, but it must not replace token-prefix identity.

Files:

- `infer/src/backend/metal/runtime.rs`
- `infer/src/backend/metal/request_state/tests.rs`

Acceptance:

- Unit test covers disk import followed by in-memory insertion.
- Unit test covers corrupt/wrong-model snapshot removal.

### Phase 1: Completed-Session Snapshot MVP

Add a completed snapshot export path that captures the final live KV+GDR state,
not only `prompt_cursor`.

New API shape:

```rust
MetalRequestState::export_qwen35_live_session_snapshot(
    token_ids: Vec<u32>,
    block_size: usize,
) -> Result<Option<Qwen35PrefixSnapshot>>
```

Contract:

- `token_ids.len()` must equal the driver's live `cache_len`.
- For a normal completed request, `token_ids = prompt_tokens + generated_token_ids`
  up to the currently materialized `cache_len`. The final sampled token can be
  absent because the standard decode loop queues the next-token forward pass one
  step behind the token returned to the client.
- Export is allowed only when the Qwen3.5/Qwen3.6 C++ session can be drained.
- Export must not truncate because Qwen3.6 GDR recurrent state cannot be
  rewound without replay.

Required code changes:

- Add `generated_token_ids: Vec<u32>` to `ActiveMetalRequest`.
- In `ActiveMetalRequest::process_token()`, push every sampled token into
  `generated_token_ids` before stop-processor buffering.
- Keep `pending_token_ids` unchanged for streaming.
- Add `ActiveMetalRequest::session_tokens()` returning
  `prompt_tokens + generated_token_ids`.
- Add `export_qwen35_live_session_snapshot()` in `request_state.rs`.
- Add `publish_completed_session_prefix()` in `MetalLivePrefixRuntime`.
- Call it from `finalize_request()` and `finalize_detached_request()` before
  request cleanup invalidates the live driver.

Files:

- `infer/src/backend/metal/runtime.rs`
- `infer/src/backend/metal/request_state.rs`
- `infer/src/backend/metal/request_state/tests.rs`

Acceptance:

- A synthetic Qwen3.5 snapshot test proves `prompt + generated` token ids are
  encoded and decoded unchanged.
- A request-state test proves snapshot export rejects mismatched
  `token_ids.len() != cache_len`.
- Existing streaming token-id semantics stay unchanged: final response
  `response_token_ids` still equals every generated token.

### Phase 2: Persist Policy For Completed Sessions

Reuse the existing SSD gate, but evaluate it on completed-session snapshots.

Persist when:

```text
saved_prefill_us > readback_us_per_token * token_count * safety
```

Where:

- `saved_prefill_us` starts as the original `first_token_at - admitted_at`.
- For a completed session, add a configurable continuation bonus because the
  next turn often extends the full transcript. Initial default:
  `continuation_bonus = 1.0`, no magic optimism.
- Keep the current `INFER_METAL_PREFIX_READBACK_US_PER_TOKEN` and
  `INFER_METAL_PREFIX_PERSIST_SAFETY`.

Do not add per-token write flush. MVP writes one full completed snapshot after
request finish.

Files:

- `infer/src/backend/metal/runtime.rs`

Acceptance:

- Short prompts still fail the persist gate.
- Long prompts / completed sessions submit one async writer job.
- Pending writer bytes stay bounded by `INFER_METAL_PREFIX_SSD_PENDING_BYTES`
  and disk `max_bytes`.

### Phase 3: Import Semantics For Session Snapshots

The existing lookup already supports "stored key is a strict prefix of the new
prompt":

Status as of 2026-06-02:

- Implemented for strict token-prefix extensions: restarted SSD cache hit
  imported 45 tokens and prefetched only the 12-token suffix in a raw
  `/v1/completions` smoke.
- Exact-prompt reuse is intentionally still not imported. The Qwen3.5 prefill
  state machine must run a terminal prompt step to sample the first token; exact
  import needs a separate imported-prefill-to-decode transition.
- OpenAI Chat history with `enable_thinking:false` is a separate product
  semantic issue. The current request appends a hidden non-thinking prefix at
  the active assistant generation position, while user-supplied prior assistant
  history usually contains only visible text. Raw token-prefix identity is not
  guaranteed across those two renderings.

- Memory: `lookup_longest_prefix()`
- Disk: `lookup_longest_disk_prefix()`

Keep that behavior. The new completed-session snapshot is just a longer key.

On import:

- Prefer the longest hit by token length.
- If disk hit wins, read/decode/import once.
- Promote the imported snapshot into memory via existing `insert_snapshot()`.
- Prefill only the suffix after `matched_len`.

Required cleanup:

- Remove `INFER_M_E13_FORCE_DISK` after the asymmetry diagnosis is closed.
- Keep `INFER_M_E13_TRACE` or replace it with structured metrics.

Files:

- `infer/src/backend/metal/runtime.rs`

Acceptance:

- Two-turn local chat smoke:
  1. Send prompt A, generate N tokens, persist completed session.
  2. Restart server.
  3. Send prompt A plus the generated transcript plus new user tail.
  4. Verify `reused_tokens` equals the completed session length and TTFT is
     reduced vs cold.

### Phase 4: Metrics And Debuggability

Add stable counters instead of relying only on env logs.

Metrics to expose:

- `metal_prefix_memory_hit`
- `metal_prefix_disk_hit`
- `metal_prefix_reused_tokens`
- `metal_prefix_disk_read_us`
- `metal_prefix_disk_decode_us`
- `metal_prefix_import_us`
- `metal_prefix_disk_write_us`
- `metal_prefix_persist_skipped_gate`
- `metal_prefix_persist_skipped_pending`
- `metal_prefix_persist_evicted_bytes`

Files:

- `infer/src/metrics.rs`
- `infer/src/backend/metal/runtime.rs`
- `infer/src/server_engine/types.rs` if telemetry projection needs new fields.

Acceptance:

- `/v1/stats` or engine telemetry shows hit counts and latencies.
- Benchmark wins entry can cite counters without enabling trace env vars.

### Phase 5: Value-Aware Eviction

Replace pure LRU with a simple value score once MVP correctness is proven:

```text
score = hit_count * saved_prefill_us / payload_bytes
```

Keep LRU as tie-breaker and as fallback for missing metadata.

Additional metadata in `MetalQwen35DiskPrefix`:

- `hit_count`
- `saved_prefill_us`
- `payload_len`
- optional `session_id`
- `created_tick`

Files:

- `infer/src/backend/metal/runtime.rs`

Acceptance:

- Unit test proves low-value short snapshot is evicted before a high-value long
  snapshot when disk budget is tight.
- Same total bytes under high/low watermarks as current LRU.

### Phase 6: Delta/Chunk Format, Only After MVP

MVP writes complete snapshots. That is acceptable because completed-session
persist is off the critical path and bounded by disk budget.

Only after MVP is benchmarked, consider a v2 format:

```text
base snapshot + append chunk(s)
```

Chunk rules:

- Default chunk target: 4 MiB. This aligns with measured good read/write
  throughput and avoids small-block random IO.
- A chunk must include enough GDR state to restore the exact end state.
- If GDR state cannot be represented as append-only, keep full completed
  snapshots and do not ship delta chunks.
- Never require reading many small chunks on request admission. If a chain
  grows past a small fixed count, compact it into a full snapshot in the
  background.

Files:

- `infer/src/backend/metal/request_state.rs`
- `infer/src/backend/metal/runtime.rs`
- Possibly a new `infer/src/backend/metal/prefix_snapshot.rs` if the format
  grows too large for `request_state.rs`.

Acceptance:

- Delta import matches full-snapshot import token-for-token.
- Chain read/import remains below the measured one-shot full snapshot path.
- Full snapshot fallback remains available.

## Verification Matrix

### Unit Tests

Run:

```bash
cargo test -p infer --no-default-features --features metal --bin metal_serve qwen35_prefix_snapshot -- --nocapture
cargo test -p infer --no-default-features --features metal --bin metal_serve kv_disk -- --nocapture
```

Required new coverage:

- Completed-session token vector is retained separately from streaming
  `pending_token_ids`.
- Completed-session snapshot export rejects token/cache length mismatch.
- Disk import promotes to memory.
- Disk budget evicts under high/low watermarks.
- Value-aware eviction, if Phase 5 lands.

### Release Build

Run:

```bash
cargo build -p infer --release --no-default-features --features metal --bin metal_serve
```

### Local Performance Smoke

Use canonical Qwen3.6:

```bash
target/release/metal_serve \
  --model-path mlx-community/Qwen3.6-35B-A3B-4bit \
  --port 9088 \
  --max-running-requests 1 \
  --max-batch-tokens 12288
```

Smoke cases:

1. Cold long prompt, disk cache empty.
2. Same-process warm extension, should hit memory snapshot.
3. Restart server, same extension, should hit SSD snapshot.
4. Disk budget pressure, should evict and continue serving.

Report:

- TTFT cold vs memory-hit vs disk-hit.
- `read_us + decode_us + import_us`.
- `write_us`, payload bytes, and skipped persist reasons.
- Cache directory size before/after.
- No process-level RSS chart; use `vmmap` / Metal footprint only if memory is
  part of the claim.

### README / Wins

If code changes land under `infer/src/backend/metal`, add a wins entry. README
should only change after the two-turn/restart smoke proves the user-visible
TTFT win.

## Rollout Order

1. Phase 1 completed-session snapshot MVP.
2. Phase 2 persist gate reuse and async write.
3. Phase 3 restart import smoke.
4. Phase 4 metrics.
5. Benchmark and wins entry.
6. Phase 5 value-aware eviction.
7. Phase 6 delta/chunk format only if full snapshots become the measured
   bottleneck.

## Kill Criteria

Stop and redesign if any of these are observed:

- Completed-session export requires replaying the prompt or generated tokens on
  the serving thread.
- Snapshot import cannot restore GDR state exactly.
- Disk hit TTFT is not meaningfully lower than cold prefill at 4k/8k/12k.
- Completed snapshot persist causes TPOT jitter while decode is active.
- Disk cache growth cannot stay within the 20 GiB default budget.

## Expected Default

Keep this default:

```text
active KV: memory resident
memory snapshot cache: auto-budgeted
SSD snapshot cache: default on, bounded 20 GiB
SSD writes: async, no per-block fsync by default
SSD reads: admission-time only, one-shot import
```

This uses SSD where the measurement says it is strong: bulk one-shot reuse and
background persistence. It avoids the path the measurement killed: active KV
paging during decode.
