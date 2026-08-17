# Ring prefill NCCL send/recv deadlock on comm_stream — CUDA, 2026-08-17

## Context

T3.2b 2D KV ownership sharding (world=4, attn_tp=2, cp=2). The ring prefill
posts NCCL recv at hop start (into the idle KV pair) and send at hop end
(of the current pair), both on `comm_stream`. The first needle-ladder run
deadlocked at tick #4 (first prefill): all 4 ranks stuck in NCCL
collectives, coordinator tore down after 120s.

## Root Cause

NCCL send/recv on the same stream deadlock when both ranks post recv
before send (NCCL docs: "may deadlock"). Stream serialization creates a
circular dependency:

- Rank 0's recv blocks `comm_stream`, waiting for rank 1's send
- Rank 1's send is behind rank 1's recv on the same stream
- Rank 1's recv blocks `comm_stream`, waiting for rank 0's send
- Rank 0's send is behind rank 0's recv on the same stream

Neither rank's stream can reach its send, so neither recv can complete.

## Fix

Add `comm_send_stream` to `DeviceContext`. P2P sends go on the send
stream; recvs stay on `comm_stream`. Two fences close the gap:

- `comm_send_waits_for_compute`: send stream waits for compute-produced
  KV buffers before sending (replaces `comm_waits_for_compute` at the
  send site).
- `comm_waits_for_comm_send`: recv stream waits for the prior hop's
  send to finish before the recv reuses the ping-pong buffer.

GDN relay (linear attention) adds `comm_send_waits_for_compute`
alongside the existing `comm_waits_for_compute` (broadcast stays on
`comm_stream`).

## Rule

NCCL P2P send and recv that can form a circular dependency (ring,
all-to-all) must run on separate streams. Collectives (all-reduce,
broadcast) are safe on a single stream because every rank enters them
unconditionally.
