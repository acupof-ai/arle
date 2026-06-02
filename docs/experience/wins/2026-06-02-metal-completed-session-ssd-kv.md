# Metal Completed-Session SSD KV Snapshot

## Goal

Make Metal Qwen3.6 SSD KV caching useful for session continuations, not only
the original prompt. Active decode KV stays resident; SSD is used only as a
bounded snapshot tier that is imported once at request admission.

## Hypothesis

If the completed request publishes the longest materialized
`prompt + generated` prefix, a later request whose tokenized prompt strictly
extends that prefix can restart the server, import the SSD snapshot, and prefill
only the unmatched suffix.

## Params

- Host: Apple M4 Pro, macOS 26.3.1
- Model: `mlx-community/Qwen3.6-35B-A3B-4bit`
- Binary: `target/release/metal_serve`
- Flags: `--max-running-requests 1 --max-batch-tokens 512 --warmup 0`
- SSD KV: `/tmp/arle-metal-kv-smoke`, `--kv-disk-max-bytes 2147483648`
- Memory snapshot tier: `--kv-memory-max-bytes 536870912`
- Trace: `RUST_LOG=info INFER_M_E10_TRACE=1 INFER_M_E13_TRACE=1`

## Results

Implemented:

- `ActiveMetalRequest` now retains `generated_token_ids` separately from the
  streaming `pending_token_ids`.
- Request completion publishes a completed-session snapshot after the final
  response delta and before request cleanup.
- Snapshot export uses the live Qwen3.5/Qwen3.6 `cache_len`, so it snapshots
  only materialized KV. In the smoke, a 64-token prompt plus 4 generated tokens
  produced a 67-token session snapshot; the final sampled token was not yet
  materialized, as expected.
- SSD persist policy now keeps strict session extensions even when they extend
  a previous cache hit.

Smoke evidence:

```text
First chat request:
  HTTP 200
  usage: prompt_tokens=64 completion_tokens=4
  prompt snapshot: tokens=64 payload=66 MiB
  completed-session snapshot: tokens=67 payload=66 MiB

Restarted raw prefix-extension request:
  SSD index: 5 entries, 348168300 bytes
  disk_match_len=45
  read_us=9225 decode_us=34835 import_us=41 imported=true
  prompt_tokens=57, resume_prefill_tokens=12
  request prefix skip_rate=0.789473
```

The SSD read/write path is therefore proven for strict token-prefix extension
across a restart.

## Problems

Exact-prompt reuse is not enabled. The current Qwen3.5 prefill state machine
needs a terminal prompt forward pass to sample the first token, so exact import
would need a new imported-prefill-to-decode transition.

OpenAI Chat history with `enable_thinking:false` did not hit the session
snapshot in this smoke. The active request appends a hidden non-thinking prefix
at the current assistant generation point; later user-provided history normally
contains only visible assistant text. That makes raw token-prefix identity
different even when the visible transcript looks like a continuation.

## Learnings

The production-safe SSD KV policy is:

- Import SSD snapshots only at request admission.
- Promote successful disk imports into the memory LRU.
- Decode entirely in resident memory.
- Publish completed-session snapshots after the client-visible final delta.
- Treat snapshot keys as exact token prefixes, not semantic chat histories.
