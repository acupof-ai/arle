# Bench Stub — KV Tier Store Dedupe

## Context

This tranche added scheduler-side inflight dedupe for the write path:

- one `(fingerprint, target)` store submission per live `StoreTicket`
- additional blocks waiting on the same durable payload now join that ticket
  instead of enqueueing duplicate disk/remote writes
- runtime store completion/failure now fans out to every block waiting on the
  deduped ticket

## What Worked

- Release `no-cuda` `cargo check`, coordinator tests, scheduler tests, and
  clippy all passed after the dedupe wiring.
- The live write path still uses one queue/event surface (`Store*`) while now
  matching the plan's dedupe requirement more closely.

## Rule

Status: `pending-remote`

Remote CUDA / guidellm validation is still required because this changes the
live scheduler store path under `infer/src/scheduler/cuda/{core,runtime}.rs`.
