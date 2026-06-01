# Streaming Tool Calls And Metal SSD Queue Bound

## Context

Review of the last three days of commits found three shipped-risk gaps:

- streaming chat tool-call extraction accepted only JSON payloads, while the
  non-streaming parser already accepted Qwen3.6 native XML
  `<function=...><parameter=...>` tool calls;
- `/v1/chat/completions` streaming post-processed tool-call blocks whenever
  tools were present, even when the request set `tool_choice="none"`;
- Metal prefix-cache SSD persist used an unbounded writer channel carrying
  already-encoded snapshot payloads. The 2026-06-01 M_e.14 evidence showed a
  single long-context snapshot payload at about 290 MB, so bursty persists could
  accumulate memory outside the disk-cache byte budget before the writer
  reserved capacity.

## What Worked

- Reused the same native XML payload parser shape on the streaming path instead
  of adding a second wire format.
- Made streaming chat route through tool-call extraction only when tools are
  effectively enabled, not merely present on the wire. `tool_choice="none"` now
  keeps the prompt and stream post-processing aligned.
- Added a pending-byte guard to the Metal SSD writer. The runtime reserves the
  estimated payload bytes before `encode_for_disk`; failed encode releases the
  reservation; actual-size drift is adjusted before the job is queued; the
  writer releases the bytes after processing the payload. Default queue budget:
  1 GiB, tunable with `INFER_METAL_PREFIX_SSD_PENDING_BYTES`.

## Verification

```text
cargo test -p chat streaming_tool_calls -- --nocapture
  6 passed

cargo test -p infer --no-default-features --features no-cuda \
  chat_completion_streaming -- --nocapture
  4 passed

cargo test -p infer --no-default-features --features metal,no-cuda \
  disk_write_handle_bounds_pending_payload_bytes -- --nocapture
  1 passed

cargo clippy -p chat -- -D warnings
  passed

cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings
  passed

cargo clippy -p infer --no-default-features --features metal,no-cuda -- -D warnings
  passed

git diff --check
  passed
```

No throughput benchmark is claimed here. This is a correctness and memory-bound
fix; the relevant acceptance gate is the targeted parser/API/queue regression
coverage above. A full Qwen3.6 Metal bench should be run before using this entry
as evidence for any latency or throughput claim.

`cargo check -p infer --no-default-features --features cuda,no-cuda` was also
attempted on the local Mac host, but stopped in `cudarc`'s build script before
ARLE typechecking because `nvcc` is not installed in this environment.

## Rule

Streaming and non-streaming tool-call parsers must support the same model output
formats. Async persistence queues carrying large encoded payloads need a byte
budget before encode/submit, not only an on-disk capacity budget after dequeue.
