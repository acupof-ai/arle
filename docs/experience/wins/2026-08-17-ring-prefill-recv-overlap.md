# Ring-prefill KV recv/compute overlap, 8 to 2 fences per hop — CUDA, 2026-08-17

> Status: pending-remote

## Goal

Cut CP ring-prefill step latency. Each hop rotates the KV pair one rank
around the ring; the rotation was a full fence bracket (compute waits comm,
comm waits compute) around a synchronous D2D rotate, serializing recv behind
the prior hop's compute. Target: overlap recv(N) with compute(N) so the
rotation adds no latency beyond the collective itself.

## Hypothesis

Two KV pairs (A=k0/v0, B=k1/v1) ping-pong. At hop N, post the recv into the
IDLE pair at the hop start (unfenced `attn_cp_recv`), then compute on the
CURRENT pair while the recv lands; send the current pair after one
`comm_waits_for_compute`. A single `compute_waits_for_comm` at the hop start
(hop > 0) covers the prior hop's recv. Fences per hop drop from 8 to 2, and
the recv latency hides under the local FA3/scatter compute.

## Parameters

```bash
# CP prefill step time at matched seq_len, world=4 (attn_tp=2, cp=2):
#   treatment (this commit) vs baseline (parent, synchronous rotate)
# Measure per-hop wall via NVTX or the engine's step timer; TTFT at
# concurrency 1 on a long-prompt workload (e.g. 131072-token prefill).
```

- Baseline: `f53107d3b^` (synchronous rotate, 8 fences/hop)
- Treatment: `f53107d3b` (ping-pong overlap, 2 fences/hop)
- Trials: 3

## Environment

- Host / GPU: 8×H20 pod (sm_90), world=4 (attn_tp=2, cp=2)
- Driver / CUDA: TBD
- Model / dtype: Qwen3.5/3.6 hybrid, BF16 KV pool
- TP / EP / slots / KV: attn_tp=2, cp=2
- Server flags: 2D engaged (world ≥ 4, attn_tp ≥ 2, cp ≥ 2)

## Results

| arm | prefill step ms | TTFT ms c=1 | delta |
|---|---:|---:|---|
| baseline | | | — |
| treatment | | | |

Raw artifacts: TBD.

## Problems

None yet. The recv-into-idle-pair posting assumes the collective's
`recv_peer` is fixed for the hop (ring neighbor), which holds for the
attn_cp ring group.

## Learnings

pending-remote. The two-pair ping-pong is the standard ring-attention overlap
shape; the fence count is the measurable, not the pair count.
