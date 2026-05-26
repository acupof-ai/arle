# DSv4 native DeepEP — parent↔child IPC roundtrip measurement

## Context

`docs/plans/2026-05-26-dsv4-deepep-process-per-rank.md` phase 1 set a
"IPC round-trip < 5 us per layer" budget for the sidecar transport,
based on intuition rather than measurement. This entry replaces that
intuition with an empirical number and locks the sidecar language
choice (C++ over Python) for phase 1 implementation.

## What worked

Three IPC channel variants benchmarked on the 8xH20 pod between a
Python parent and a Python child process, 2000 warmed iterations each:

| Channel | min | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| JSON over stdio | 31.7 us | 33.6 us | 37.7 us | 48.1 us | 988 us |
| Binary (16 B) over stdio | 10.5 us | 10.8 us | 10.9 us | 19.4 us | 28.0 us |
| Binary (16 B) over anonymous pipes | 9.9 us | 10.3 us | 10.4 us | 18.6 us | 25.3 us |

Key observations:

- JSON parse + serialize alone adds ~22 us per call. Killing JSON saves
  more than 2x.
- Anonymous pipes are essentially tied with stdio in Python — the OS
  pipe primitive bottoms at ~10 us when crossing a Python interpreter.
- The 5 us design-doc budget is not reachable with any Python
  parent↔Python child variant. To meet it would require either
  shared-memory + futex or eventfd in a C/C++ binary.

### Reframed budget

The 5 us number came from "make IPC negligible vs DeepEP itself".
The actually-binding budget is "per-layer overhead < (NCCL-DeepEP
gap)". From the binding-constraints table (L6) the NCCL DeepEP-style
fallback eats ~20 ms / rank-range per token; over the DSv4 ~61 layers
that is ~330 us / layer. Tuned DeepEP itself runs ~78 us / layer
(42 us dispatch + 36 us combine on this pod). So the realistic
budget for sidecar overhead is **~250 us per layer**, not 5 us.

At p99 19 us per pipe roundtrip, Python sidecar IPC consumes only 8%
of the budget — comfortable.

## Architectural decision

Phase 1 still goes with a **C++ sidecar**, not Python, on the user's
"strict no-Python" preference. Reasoning beyond the budget:

- CLAUDE.md "no Python on the hot path" rule applies in spirit even
  when Python lives in a sidecar subprocess.
- C++ sidecar links against `deep_ep_cpp.so` directly (the same
  extension Python imports) and bypasses interpreter dispatch on
  every layer call.
- C++ control plane can use `eventfd` + shared-memory ring buffer
  for ~1–3 us roundtrip, well under both the 5 us aspirational and
  the 250 us binding budget.
- Maintenance cost is offset by removing Python from the production
  data plane entirely.

The Python sidecar path remains the documented fallback if C++
integration hits a structural blocker.

## Artifacts

- IPC bench script + summary: `dsv4-deepep-ipc-roundtrip-20260526/`

## Rule

Per-layer IPC budgets need empirical measurement before they shape
architecture decisions. A 5 us target with no measurement is
indistinguishable from "as fast as possible"; a 250 us target backed
by the NCCL-DeepEP gap lets a Python sidecar pass and stays honest
about what we're trading.
